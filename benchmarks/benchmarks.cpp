#include <iostream>
#include <string>
#include <vector>

#include "measureTime.hpp"

#include "anyfold.hpp"

template<typename T>
T uniformRandom(T const& min = T(0), T const& max = T(1))
{
	static std::default_random_engine generator;
	std::uniform_real_distribution< T > distribution(min, max);
	return distribution(generator);
}

template<typename T>
T* genRandomArray(size_t size, T min, T max)
{
	T* array = new T[size];
	for (size_t i = 0; i < size; ++i)
	{
		array[i] = uniformRandom(min, max);
	}

	return array;
}

void warmUpOpenCL()
{
	float* image = genRandomArray(8*8*8, 0.0f, 1.0f);
	int imageShape[] = {8, 8, 8};
	float* kernel = genRandomArray(3*3*3, 0.0f, 1.0f);
	int kernelShape[] = {3, 3, 3};
	float* output = new float[8*8*8];
	anyfold::opencl::convolve_3d(image, imageShape, kernel, kernelShape, output);

	delete[] image;
	delete[] kernel;
	delete[] output;
}


int main(int argc, char *argv[])
{
	warmUpOpenCL();

	int ks = 3;
	SimpleTimer timer;
	int rep = 32;
	for (int i = 32; i <= 256; i *= 2)
	{
		std::cout << "\nImage size:  " << i
		          << "\nKernel size: " << ks << std::endl;
		
		float* image = genRandomArray(i*i*i, 0.0f, 1.0f);
		int imageShape[] = {i, i, i};
		float* kernel = genRandomArray(ks*ks*ks, 0.0f, 1.0f);
		int kernelShape[] = {ks, ks, ks};
		float* output = new float[i*i*i];

		uint64_t time = 0;
		// OpenCL
		for(int r = 0; r < rep; ++r)
		{
			timer.start();
			anyfold::opencl::convolve_3d(image, imageShape, kernel, kernelShape, output);
			timer.end();
			// std::cout << "OpenCL: \n";
			// timer.print(true);
			time += timer.getNS();
		}
		std::cout << "OpenCL (average): \n" << (float)time / 1000000000.0f / (float)rep << std::endl;

		// CPU
		float* outputCPU = new float[i*i*i];
		timer.start();
		anyfold::cpu::convolve_3d(image, imageShape, kernel, kernelShape, outputCPU);
		timer.end();
		std::cout << "CPU (single core): \n";
		timer.print(true);

		delete[] image;
		delete[] kernel;
		delete[] output;
		delete[] outputCPU;
	}	
	return 0;
}
