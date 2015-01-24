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

int main(int argc, char *argv[])
{
	if(argc != 3 )
	{
		std::cerr << "Please provide a convolution method\n"
		          << "<0> : Buffer\n"
		          << "<1> : Buffer and local memory\n"
		          << "<2> : Images\n"
		          << "<3> : Images and local memory\n"
		          << "\n and a flag for CPU testing!\n"
		          << "<0> : CPU comparision off\n"
		          << "<1> : CPU comparision on" << std::endl;
		exit(-1);
	}
	int method = std::atoi(argv[1]);
	bool meas_cpu = std::atoi(argv[2]);

	//print header of table
	std::cout << "image size;kernel size;opencl;cpu" << std::endl;

	SimpleTimer timer;
	int rep = 10; //number of repeats
	//kernels
	std::vector<std::vector<int>> k = {{  3,   9,  15},
	                                   {  9,  21,  27},
	                                   { 17,  17,  29},
	                                   { 27,  39,  51},
	                                   { 39,  51,  21},
	                                   { 23,  81,  85},
	                                   { 59,  59,  59},
	                                   { 69,  69,  69},
	                                   { 53, 127, 109}};

	//for every kernel
	for(auto ks : k)
	//for different image sizes (height, width)
	for(int i = 128; i <= 1024; i += 64)
	//for different 3D-image depth
	for(int z = 100; z <= 600; z += 100)
	{
		std::cout << i << "x" << i << "x" << z << ";"
		          << ks[0] << "x" << ks[1] << "x" << ks[2] <<";";

		float* image = genRandomArray(i*i*z, 0.0f, 1.0f);
		int imageShape[] = {i, i, z};
		float* output = new float[i*i*z];

		float* imagePadded = genRandomArray((i+ks[0]/2)*
		                                    (i+ks[1]/2)*
		                                    (z+ks[2]/2),
		                                    0.0f, 1.0f);
		int imageShapePadded[] = {i+ks[0]/2, i+ks[1]/2, z+ks[2]/2};
		float* outputPadded = new float[(i+ks[0]/2)*
		                                (i+ks[1]/2)*
		                                (z+ks[2]/2)];

		float* kernel = genRandomArray(ks[0]*ks[1]*ks[2], 0.0f, 1.0f);
		int kernelShape[] = {ks[0], ks[1], ks[2]};

		uint64_t time = 0;
		// OpenCL
		for(int r = 0; r < rep; ++r)
		{
			using namespace anyfold::opencl;
			timer.start();
			switch(method)
			{
			case 0:
				convolve_3dBuffer(imagePadded,
				                  imageShapePadded,
				                  kernel, kernelShape,
				                  outputPadded);
				break;
			case 1:
				convolve_3dBufferLocalMem(imagePadded,
				                          imageShapePadded,
				                          kernel, kernelShape,
				                          outputPadded);
				break;
			case 2:
				convolve_3dImage(image, imageShape,
				                 kernel, kernelShape, output);
				break;
			case 3:
				convolve_3dImageLocalMem(image, imageShape,
				                         kernel, kernelShape,
				                         output);
				break;
			default:
				std::cerr << "Invalid method!" << std::endl;
				exit(-1);
			}
			timer.end();
			time += timer.getNS();
		}
		std::cout << (float)time / 1000000000.0f / (float)rep << ";"
		          << std::flush;

		// CPU
		if(meas_cpu)
		{
			float* outputCPUPadded = new float[(i+ks[0]/2)*
			                                   (i+ks[1]/2)*
			                                   (z+ks[2]/2)];
			timer.start();
			anyfold::cpu::convolve_3d(imagePadded, imageShapePadded,
			                          kernel, kernelShape,
			                          outputCPUPadded);
			timer.end();
			timer.print();
			delete[] outputCPUPadded;
		}

		std::cout << std::endl;

		delete[] image;
		delete[] imagePadded;
		delete[] kernel;
		delete[] output;
		delete[] outputPadded;
	}
	return 0;
}

