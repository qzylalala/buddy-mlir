#ifndef TIME_H
#define TIME_H

#include <stdio.h>

#include <time.h>
#include <sys/time.h>

// return us
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec * 1e6  + (double)tp.tv_usec);
}

#endif // TIME_H