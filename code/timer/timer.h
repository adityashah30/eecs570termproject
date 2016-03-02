#include <unistd.h>
#include <sys/time.h>

#define USECS_IN_SEC 1000000

class Timer
{
private:
    struct timeval start, stop;
    long long elapsedTime; //ElapsedTime in microseconds
public:
    Timer();
    void startTimer();
    void stopTimer();
    long long getElapsedTime();
};
