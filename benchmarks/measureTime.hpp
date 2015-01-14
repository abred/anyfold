#ifndef _MEASURETIME_HPP_
#define _MEASURETIME_HPP_

#include <time.h>

class SimpleTimer
{
public:
	void start()
	{
		clock_gettime(CLOCK_REALTIME, &startTime);
	};
	void end()
	{
		clock_gettime(CLOCK_REALTIME, &endTime);
	};
	uint64_t getNS()
	{
		struct timespec t = timeDiff(startTime, endTime);
		return t.tv_sec * 1000000000 + t.tv_nsec;
	}
	void print(bool longOutput = false)
	{
		struct timespec t = timeDiff(startTime, endTime);
		if(longOutput)
		{
			printf("Zeit: %ld.%03lds\n", t.tv_sec, t.tv_nsec/1000000);
		}
		else
		{
			printf("%ld.%03lds\n", t.tv_sec, t.tv_nsec/1000000);
		}
	};

private:
	struct timespec timeDiff(struct timespec start, struct timespec end)
	{
		struct timespec temp;
		if ((end.tv_nsec-start.tv_nsec)<0) {
			temp.tv_sec = end.tv_sec-start.tv_sec-1;
			temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
		} else {
			temp.tv_sec = end.tv_sec-start.tv_sec;
			temp.tv_nsec = end.tv_nsec-start.tv_nsec;
		}
		return temp;
	};
	
	struct timespec startTime;
	struct timespec endTime;
};

#endif /* _MEASURETIME_HPP_ */
