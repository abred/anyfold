#ifndef CONVOLUTION3DCL_HPP
#define CONVOLUTION3DCL_HPP

#include <vector>
#include <string>

#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
	#include "CL/cl.h"
#endif

class Convolution3DCL
{
public:
	Convolution3DCL() = default;
	~Convolution3DCL() = default;

	bool setupCLcontext();
	void createProgramAndLoadKernel(const char* fileName, const char* kernelName);
	void convolve3D(/* something image3D, something filterkernel3D */);


private:
	void createProgram(const std::string& source);
	void loadKernel(const char* kernelName);
	void* getDeviceInfo(cl_device_id id, cl_device_info info);
	std::string getDeviceName(cl_device_id id);
	std::string getPlatformInfo(cl_platform_id id, cl_platform_info info);
	void checkError(cl_int status, const char* label, const char* file, int line);


private:
	cl_context context;
	std::vector<cl_platform_id> platforms;
	std::vector<cl_device_id> devices;

	cl_program program;
	cl_kernel kernel;

	cl_int status = CL_SUCCESS;
};


#endif /* CONVOLUTION3DCL_HPP */
