#ifndef _MEASURETIME_HPP_
#define _MEASURETIME_HPP_

#include <chrono>
#include <iostream>

class SimpleTimer
{
public:
	void start()
	{
		startTime = std::chrono::high_resolution_clock::now();
	};
	void end()
	{
		endTime = std::chrono::high_resolution_clock::now();

	};
	uint64_t getNS()
	{
		return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
	}
	void print()
	{
		auto t = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime) -
		         std::chrono::duration_cast<std::chrono::seconds>(endTime-startTime);
		std::cout << std::chrono::duration_cast<std::chrono::seconds>(endTime-startTime).count()
		          << "."
		          << t.count();
	};

private:
	std::chrono::high_resolution_clock::time_point startTime;
	std::chrono::high_resolution_clock::time_point endTime;
};

#endif /* _MEASURETIME_HPP_ */
