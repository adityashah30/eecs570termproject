#include "timer.h"

Timer::Timer()
{
    elapsedTime = 0.0f;
}

void Timer::startTimer()
{
    gettimeofday(&start, NULL);
}

void Timer::stopTimer()
{
    gettimeofday(&stop, NULL);
    elapsedTime = (stop.tv_sec-start.tv_sec)*MSECS_IN_SEC+
                  (stop.tv_usec-start.tv_usec)/USECS_IN_MSEC;
}

long long Timer::getElapsedTime()
{
    return elapsedTime;
}
