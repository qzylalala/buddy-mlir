"""XGBoost as cost model"""

import logging
import time

from typing import Dict

import numpy as np
from contrib.popen_pool import PopenPoolExecutor, StatusKind

from .metric import get_rank, cover_curve, max_curve, recall_curve
from .model_based_tuner import CostModel, FeatureCache

try:
    from xgboost.callback import TrainingCallback
except ImportError:
    class TrainingCallback:
        pass

import xgboost as xgb


logger = logging.getLogger("autotuner")


class XGBoostCostModel(CostModel):
    """XGBoost as cost model

    Parameters
    ----------
    task: Task
        The tuning task
    loss_type: str
        If is 'reg', use regression loss to train cost model.
                     The cost model predicts the normalized flops.
        If is 'rank', use pairwise rank loss to train cost model.
                     The cost model predicts relative rank score.
        If is 'rank-binary', use pairwise rank loss with binarized labels to train cost model.
                     The cost model predicts relative rank score.
    num_threads: int, optional
        The number of threads.
    log_interval: int, optional
        If is not none, the cost model will print training log every `log_interval` iterations.
    upper_model: XGBoostCostModel, optional
        The upper model used in transfer learning
    """

    def __init__(
        self,
        task,
        loss_type="reg",
        num_threads=None,
        log_interval=25,
        upper_model=None,
    ):
        super(XGBoostCostModel, self).__init__()

        self.task = task
        self.target = task.target
        self.space = task.config_space
        self.loss_type = loss_type
        self.num_threads = num_threads
        self.log_interval = log_interval

        if loss_type == "reg":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "reg:linear",
            }
        elif loss_type in ("rank", "rank-binary"):
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "rank:pairwise",
            }
        else:
            raise RuntimeError("Invalid loss type: " + loss_type)

        self.xgb_params["verbosity"] = 0
        if num_threads:
            self.xgb_params["nthread"] = num_threads
        self.bst = None

        # TODO: 后续可以继续探索 feature 的选用
        self.fea_type = "knob"
        self.feature_extract_func = _extract_knob_feature_index

        if upper_model:  # share a same feature cache with upper model
            self.feature_cache = upper_model.feature_cache
        else:
            self.feature_cache = FeatureCache()
        self.upper_model = upper_model
        self.feature_extra_ct = 0
        self.pool = None
        self.base_model = None

        self._sample_size = 0
        self._reset_pool(self.space, self.target, self.task)

    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""

        if self.upper_model:  # base model will reuse upper model's pool,
            self.upper_model._reset_pool(space, target, task)
            return

        self._close_pool()

        self.pool = PopenPoolExecutor(
            max_workers=self.num_threads,
            initializer=_extract_popen_initializer,
            initargs=(space, target, task),
        )

    def _close_pool(self):
        if self.pool:
            self.pool = None

    def _get_pool(self):
        if self.upper_model:
            return self.upper_model._get_pool()
        return self.pool

    def _base_model_discount(self):
        return 1.0 / (2 ** (self._sample_size / 64.0))

    def fit(self, xs, ys, plan_size):
        tic = time.time()
        self._reset_pool(self.space, self.target, self.task)

        x_train = self._get_feature(xs)
        y_train = np.array(ys)
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)

        valid_index = y_train > 1e-6
        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])
        self._sample_size = len(x_train)

        if self.base_model:
            discount = self._base_model_discount()
            if discount < 0.05:  # discard base model
                self.base_model.upper_model = None
                self.base_model = None
            else:
                dtrain.set_base_margin(discount * self.base_model.predict(xs, output_margin=True))

        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=8000,
            callbacks=[
                CustomCallback(
                    stopping_rounds=20,
                    metric=f"tr-a-recall@{plan_size}",
                    evals=[(dtrain, "tr")],
                    maximize=True,
                    fevals=[xgb_average_recalln_curve_score(plan_size)],
                    verbose_eval=self.log_interval,
                    loss_type=self.loss_type,
                )
            ],
        )

        logger.debug(
            "XGB train: %.2f\tobs: %d\terror: %d\tn_cache: %d",
            time.time() - tic,
            len(xs),
            len(xs) - np.sum(valid_index),
            self.feature_cache.size(self.fea_type),
        )

    def fit_log(self, records, plan_size, min_seed_records=500):
        tic = time.time()

        # filter data, only pick the data with a same task
        data = []
        for inp, res in records:
            if inp.task.name == self.task.name:
                data.append((inp, res))

        logger.debug("XGB load %d entries from history log file", len(data))

        # extract feature
        self._reset_pool(self.space, self.target, self.task)
        pool = self._get_pool()
        # TODO: check features
        # if self.fea_type == "itervar":
        #     feature_extract_func = _extract_itervar_feature_log
        if self.fea_type == "knob":
            feature_extract_func = _extract_knob_feature_log
        # elif self.fea_type == "curve":
        #     feature_extract_func = _extract_curve_feature_log
        else:
            raise RuntimeError("Invalid feature type: " + self.fea_type)
        result = pool.map_with_error_catching(feature_extract_func, data)
        result = list(result)  # store results so we can iterate through them twice

        # get maximum feature length
        fea_len = -1
        for res in result:
            if res.status != StatusKind.COMPLETE:
                continue
            x, _ = res.value
            fea_len = max(fea_len, x.shape[0])

        xs, ys = [], []
        for res in result:
            if res.status != StatusKind.COMPLETE:
                continue
            x, y = res.value
            # Features may not be the same size, pad them until they are
            if fea_len > len(x):
                xs.append(np.pad(x, (0, fea_len - len(x))))
            else:
                xs.append(x)
            ys.append(y)

        if len(xs) < min_seed_records:  # no enough samples
            return False

        xs, ys = np.array(xs), np.array(ys)
        x_train = xs
        y_train = ys
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)

        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])

        plan_size *= 2
        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=400,
            callbacks=[
                CustomCallback(
                    stopping_rounds=100,
                    metric=f"tr-a-recall@{plan_size}",
                    evals=[(dtrain, "tr")],
                    maximize=True,
                    fevals=[xgb_average_recalln_curve_score(plan_size)],
                    verbose_eval=self.log_interval,
                    loss_type=self.loss_type,
                )
            ],
        )

        logger.debug("XGB train: %.2f\tobs: %d", time.time() - tic, len(xs))

        return True

    def predict(self, xs, output_margin=False):
        feas = self._get_feature(xs)
        dtest = xgb.DMatrix(feas)

        if self.base_model:
            dtest.set_base_margin(
                self._base_model_discount() * self.base_model.predict(xs, output_margin=True)
            )

        return self.bst.predict(dtest, output_margin=output_margin)

    def load_basemodel(self, base_model):
        self.base_model = base_model
        self.base_model._close_pool()
        self.base_model.upper_model = self

    def spawn_base_model(self):
        return XGBoostCostModel(
            self.task, self.fea_type, self.loss_type, self.num_threads, self.log_interval, self
        )

    def _get_feature(self, indexes):
        """get features for indexes, run extraction if we do not have cache for them"""
        # free feature cache
        if self.feature_cache.size(self.fea_type) >= 100000:
            self.feature_cache.clear(self.fea_type)

        fea_cache = self.feature_cache.get(self.fea_type)

        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in fea_cache]

        if need_extract:
            pool = self._get_pool()
            feas = pool.map_with_error_catching(self.feature_extract_func, need_extract)
            for i, fea in zip(need_extract, feas):
                fea_cache[i] = fea.value if fea.status == StatusKind.COMPLETE else None

        feature_len = -1
        for idx in indexes:
            if fea_cache[idx] is not None:
                feature_len = max(fea_cache[idx].shape[-1], feature_len)

        ret = np.empty((len(indexes), feature_len), dtype=np.float32)
        for i, ii in enumerate(indexes):
            t = fea_cache[ii]
            if t is not None and t.shape[0] < feature_len:
                t = np.pad(t, (0, feature_len - t.shape[0]))
            ret[i, :] = t if t is not None else 0
        return ret

    def __del__(self):
        self._close_pool()


# Global variables for passing arguments to extract functions.
_extract_space = None
_extract_target = None
_extract_task = None


def _extract_popen_initializer(space, target, task):
    global _extract_space, _extract_target, _extract_task
    _extract_space = space
    _extract_target = target
    _extract_task = task


def _extract_knob_feature_index(args):
    """extract knob feature for an index in extract_space"""
    config = _extract_space.get(args)

    return config.get_flatten_feature()


def _extract_knob_feature_log(arg):
    """extract knob feature for log items"""
    inp, res = arg
    config = inp.config
    x = config.get_flatten_feature()

    if res.error_no == 0:
        # TODO: 我们现在直接默认每个任务的 FLOP 固定，暂时不支持计算任务负载的 FLOP
        # with inp.target:  # necessary, for calculating flops of this task
        #     inp.task.instantiate(config)
        y = inp.task.flop / np.mean(res.costs)
    else:
        y = 0.0
    return x, y


def _binarize_evals(evals):
    """binarize evaluation labels"""
    bin_evals = []
    for evalset in evals:
        # binarize labels in xgb.dmatrix copy
        barray = evalset[0].get_data().copy()
        blabel = evalset[0].get_label().copy()
        blabel[blabel < 0.5] = 0.0
        blabel[blabel >= 0.5] = 1.0
        # pylint: disable=R1721
        bin_evals.append(tuple([xgb.DMatrix(barray, blabel)] + [e for e in evalset[1:]]))
    return bin_evals


class XGBoostCallback(TrainingCallback):
    """Base class for XGBoost callbacks."""

    def __call__(self, env: "xgb.core.CallbackEnv"):
        # Compatibility with xgboost < 1.3
        return self.after_iteration(env.model, env.iteration, env.evaluation_result_list)

    def after_iteration(self, model: "xgb.Booster", epoch: int, evals_log: Dict):
        raise NotImplementedError


class CustomCallback(XGBoostCallback):
    """
    Callback function for xgboost.
    Support custom evaluation function and early-stopping.
    """

    def __init__(
        self,
        stopping_rounds,
        metric,
        fevals,
        loss_type="reg",
        evals=(),
        log_file=None,
        maximize=False,
        verbose_eval=True,
        skip_every=2,
    ):
        """Init function"""
        self.stopping_rounds = stopping_rounds
        self.metric = metric
        self.metric_shortname = metric.split("-")[1]
        self.fevals = fevals
        self.evals = evals
        self.log_file = log_file
        self.maximize = maximize
        self.verbose_eval = verbose_eval
        self.loss_type = loss_type
        self.skip_every = skip_every
        self.state = {}

    def after_iteration(self, model: "xgb.Booster", epoch: int, evals_log: Dict):
        """Run after each iteration.  Return True when training should stop."""
        # pylint:disable = import-outside-toplevel
        try:
            from xgboost.callback import _fmt_metric  # type: ignore
        except ImportError:
            # Compatibility with xgboost >= 1.6
            def _fmt_metric(value, show_stdv=True):
                """format metric string"""
                if len(value) == 2:
                    return f"{value[0]}:{value[1]:.5f}"
                if len(value) == 3:
                    if show_stdv:
                        return f"{value[0]}:{value[1]:.5f}+{value[2]:.5f}"
                    return f"{value[0]}:{value[1]:.5f}"
                raise ValueError("wrong metric value", value)

        ##### init state #####
        if not self.state:
            self.state["maximize_score"] = self.maximize
            self.state["best_iteration"] = 0
            if self.maximize:
                self.state["best_score"] = float("-inf")
            else:
                self.state["best_score"] = float("inf")

            assert model is not None
            if model.attr("best_score") is not None:
                self.state["best_score"] = float(model.attr("best_score"))
                self.state["best_iteration"] = int(model.attr("best_iteration"))
                self.state["best_msg"] = model.attr("best_msg")
            else:
                model.set_attr(best_iteration=str(self.state["best_iteration"]))
                model.set_attr(best_score=str(self.state["best_score"]))
        res_dict = {}

        if epoch % self.skip_every == 1:
            return False

        ##### evaluation #####
        mod_evals = self.evals
        if self.loss_type == "rank-binary":
            mod_evals = _binarize_evals(self.evals)

        if self.loss_type == "rank" and int(xgb.__version__[0]) >= 2:
            # since xgboost pr#8931
            raise RuntimeError(
                "Use 'rank-binary' instead of 'rank' loss_type with xgboost %s >= 2.0.0"
                % xgb.__version__
            )

        for feval in self.fevals:
            bst_eval = model.eval_set(mod_evals, epoch, feval)
            res = [x.split(":") for x in bst_eval.split()]
            for kv in res[1:]:
                res_dict[kv[0]] = [float(kv[1])]

        eval_res = []
        keys = list(res_dict.keys())
        keys.sort(key=lambda x: x if self.metric_shortname not in x else "a" + x)
        for key in keys:
            v = res_dict[key]
            eval_res.append([key] + v)

        ##### print eval result #####
        if (
            not isinstance(self.verbose_eval, bool)
            and self.verbose_eval
            and epoch % self.verbose_eval == 0
        ):
            infos = [f"XGB iter: {epoch:3d}"]
            for item in eval_res:
                if "null" in item[0]:
                    continue
                infos.append(f"{item[0]}: {item[1]:.6f}")

            logger.debug("\t".join(infos))
            if self.log_file:
                with open(self.log_file, "a") as fout:
                    fout.write("\t".join(infos) + "\n")

        ##### choose score and do early stopping #####
        score = None
        for item in eval_res:
            if item[0] == self.metric:
                score = item[1]
                break
        assert score is not None

        best_score = self.state["best_score"]
        best_iteration = self.state["best_iteration"]
        maximize_score = self.state["maximize_score"]

        if (maximize_score and score > best_score) or (not maximize_score and score < best_score):
            msg = f"[{epoch}] " + "\t".join([_fmt_metric(x) for x in eval_res])
            self.state["best_msg"] = msg
            self.state["best_score"] = score
            self.state["best_iteration"] = epoch
            # save the property to attributes, so they will occur in checkpoint.
            if model is not None:
                model.set_attr(
                    best_score=str(self.state["best_score"]),
                    best_iteration=str(self.state["best_iteration"]),
                    best_msg=self.state["best_msg"],
                )
        elif epoch - best_iteration >= self.stopping_rounds:
            best_msg = self.state["best_msg"]
            if self.verbose_eval:
                logger.debug("XGB stopped. Best iteration: %s ", best_msg)
            return True

        return False


# feval wrapper for xgboost
def xgb_max_curve_score(N):
    """evaluate max curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        scores = labels[trials]
        curve = max_curve(scores)
        return f"Smax@{N}", curve[N] / np.max(labels)

    return feval


def xgb_recalln_curve_score(N):
    """evaluate recall-n curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return f"recall@{N}", curve[N]

    return feval


def xgb_average_recalln_curve_score(N):
    """evaluate average recall-n curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return f"a-recall@{N}", np.sum(curve[:N]) / N

    return feval


def xgb_recallk_curve_score(N, topk):
    """evaluate recall-k curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks, topk)
        return f"recall@{topk}", curve[N]

    return feval