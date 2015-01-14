#ifndef CONVOLUTION3DCL_HPP
#define CONVOLUTION3DCL_HPP

#include <vector>
#include <string>

#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
	#include "CL/cl.h"
#endif

#include "image_stack_utils.h"

namespace anyfold {

namespace opencl {

class Convolution3DCL
{
public:
	Convolution3DCL() = default;
	~Convolution3DCL() = default;

	bool setupCLcontext();
	void createProgramAndLoadKernel(const char* fileName,
	                                const char* kernelName
					size_t filterSize);
	void setupKernelArgs(image_stack_cref _image,
	                     image_stack_cref _kernel,
	                     const std::vector<int>& _offset);
	void execute();
	void getResult(image_stack_ref result);
	// void convolve3D(/* something image3D, something filterkernel3D */);


private:
	void createProgram(const std::string& source, size_t filterSize);
	void loadKernel(const char* kernelName);
	void* getDeviceInfo(cl_device_id id, cl_device_info info);
	std::string getDeviceName(cl_device_id id);
	std::string getPlatformInfo(cl_platform_id id, cl_platform_info info);
	void checkError(cl_int status, const char* label,
	                const char* file, int line);


private:
	cl_context context;
	std::vector<cl_platform_id> platforms;
	std::vector<cl_device_id> devices;

	cl_program program;
	cl_kernel kernel;
	cl_command_queue queue;

	cl_int status = CL_SUCCESS;

	cl_mem inputImage;
	cl_mem outputImage;
	cl_mem filterWeightsBuffer;
	std::size_t size[3];

};

} /* namespace opencl */
} /* namespace anyfold */

#endif /* CONVOLUTION3DCL_HPP */
