#include <iostream>
#include <fstream>

#include "opencl/convolution3DCL.hpp"

namespace anyfold {

namespace opencl {

#define CHECK_ERROR(status, fctname) { \
	checkError(status, fctname, __FILE__, __LINE__ -1); \
	}

void CL_CALLBACK errorCallback(const char *errinfo,
                               const void *private_info, size_t cb,
                               void *user_data)
{
	std::cerr << errinfo << " (reported by error callback)" << std::endl;
}

std::string Convolution3DCL::getPlatformInfo(cl_platform_id id, cl_platform_info info)
{
	size_t size = 0;
	cl_int status = clGetPlatformInfo(id, info, 0, nullptr, &size);
	CHECK_ERROR(status, "clGetPlatformInfo");

	std::string result;
	result.resize(size);
	status = clGetPlatformInfo(id, info, size, const_cast<char*>(result.c_str()), nullptr);
	CHECK_ERROR(status, "clGetPlatformInfo");

	return result;
}

std::string Convolution3DCL::getDeviceName(cl_device_id id)
{
	size_t size = 0;
	cl_int status = clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);
	CHECK_ERROR(status, "clGetDeviceInfo");

	std::string result;
	result.resize(size);
	status = clGetDeviceInfo(id, CL_DEVICE_NAME, size, const_cast<char*>(result.c_str()), nullptr);
	CHECK_ERROR(status, "clGetDeviceInfo");

	return result;
}

void* Convolution3DCL::getDeviceInfo(cl_device_id id, cl_device_info info)
{
	size_t size = 0;
	cl_int status = clGetDeviceInfo(id, info, 0, nullptr, &size);
	CHECK_ERROR(status, "clGetDeviceInfo");

	void* result = malloc(size);
	status = clGetDeviceInfo(id, info, size, result, nullptr);
	CHECK_ERROR(status, "clGetDeviceInfo");

	return result;
}

void Convolution3DCL::createProgramAndLoadKernel(const char* fileName, const char* kernelName, size_t filterSize)
{
	std::string content;
	std::ifstream in(fileName, std::ios::in);
	if(in)
	{
		in.seekg(0, std::ios::end);
		content.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&content[0], content.size());
		in.close();
	}

	createProgram(content, filterSize);
	loadKernel(kernelName);
}

void Convolution3DCL::createProgram(const std::string& source, 
				    size_t filterSize)
{
	const char* sources [1] = { source.c_str() };

	program = clCreateProgramWithSource(context, 1, sources, nullptr, &status);
	CHECK_ERROR(status, "clCreateProgramWithSource");

	std::string defines = std::string("-D FILTER_SIZE=") + std::to_string(filterSize) +
	                      std::string(" -D FILTER_SIZE_HALF=") + std::to_string(filterSize/2);
	status = clBuildProgram(program, 0, nullptr,
	                        defines.c_str(),
	                        nullptr, nullptr);

	size_t logSize = 0;
	clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

	if(logSize > 1)
	{
		std::string log;
		log.resize(logSize);
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
		                      logSize, const_cast<char*>(log.c_str()),
		                      nullptr);
		std::cout << log << std::endl;
	}

	CHECK_ERROR(status, "clBuildProgram");
}

void Convolution3DCL::loadKernel(const char* kernelName)
{
	kernel = clCreateKernel(program, kernelName, &status);
	CHECK_ERROR(status, "clCreateKernel");
}

bool Convolution3DCL::setupCLcontext()
{
	cl_uint platformCount = 0;
	clGetPlatformIDs(0, nullptr, &platformCount);

	// if (platformCount == 0)
	// {
	// 	std::cerr << "No OpenCL platform found" << std::endl;
	// 	return false;
	// }
	// else
	// {
	// 	std::cout << "Found " << platformCount << " platform(s)" << std::endl;
	// }

	platforms.resize(platformCount);
	clGetPlatformIDs(platformCount, platforms.data(), nullptr);

	// for (cl_uint i = 0; i < platformCount; ++i)
	// {
	// 	std::cout << "\t (" << (i+1) << ") : "
	// 	          << getPlatformInfo(platforms[i], CL_PLATFORM_NAME)
	// 	          << "\n\tExtensions: \n"
	// 	          << "\t" << getPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS)
	// 	          << std::endl;
	// }
	// std::cout << "Platform (1) chosen" << std::endl;

	cl_uint deviceCount = 0;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, nullptr,
	                         &deviceCount);
	CHECK_ERROR(status, "clGetDeviceIDs");

	// if (deviceCount == 0)
	// {
	// 	std::cerr << "No OpenCL device found" << std::endl;
	// 	return false;
	// }
	// else
	// {
	// 	std::cout << "Found " << deviceCount << " device(s)" << std::endl;
	// }

	devices.resize(deviceCount);
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, deviceCount,
	               devices.data(), nullptr);
	CHECK_ERROR(status, "clGetDeviceIDs");

	// for (cl_uint i = 0; i < deviceCount; ++i) {
	// 	std::cout << "\t (" << (i+1) << ") : " << getDeviceName(devices[i])
	// 	          << "\n\tExtensions: \n";
	// 	char* str = (char*)getDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS);
	// 	std::cout << "\t" << str << std::endl << std::endl;
	// 	free(str);
	// }
	// std::cout << "Device (1) chosen" << std::endl;

	const cl_context_properties contextProperties [] =
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties> (platforms[0]),
		0
	};

	context = clCreateContext(contextProperties, deviceCount,
	                                     devices.data(), errorCallback, nullptr, &status);
	CHECK_ERROR(status, "clCreateContext");

	// std::cout << "Context created" << std::endl;

	queue = clCreateCommandQueue(context, devices[0],
	                             0, &status);
	CHECK_ERROR(status, "clCreateCommandQueue");

	return true;
}

void Convolution3DCL::setupKernelArgs(image_stack_cref image,
                                      image_stack_cref filterKernel,
                                      const std::vector<int>& offset)
{
	size[0] = image.shape()[0];
	size[1] = image.shape()[1];
	size[2] = image.shape()[2];
	// std::cout << size[0] << " " << size[1] << " " << size[2] << std::endl;
	// for(int i = 0; i < 10; i++){
	// 	for(int j = 0; j < 10; j++){
	// 		for(int k = 0; k < 10; k++){
	// 			std::cout << image.data()[i*100+j*10+k] << " ";
	// 		}std::cout << std::endl;
	// 	}std::cout << std::endl;
	// }std::cout << std::endl;

	// const cl_image_format format = { CL_R, CL_UNORM_INT8 };
	const cl_image_format format = { CL_R, CL_FLOAT };
	inputImage = clCreateImage3D(context,
	                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	                             &format,
	                             size[0], size[1], size[2],
	                             0, 0, const_cast<float*>(image.data()), &status);
	CHECK_ERROR(status, "clCreateImage3D");

	outputImage = clCreateImage3D(context, CL_MEM_WRITE_ONLY, &format,
	                              size[0], size[1], size[2],
	                              0, 0, nullptr, &status);
	CHECK_ERROR(status, "clCreateImage3D");

	filterWeightsBuffer = clCreateBuffer(context,
	                                     CL_MEM_READ_ONLY |
	                                     CL_MEM_COPY_HOST_PTR,
	                                     sizeof(float) * filterKernel.num_elements(),
	                                     const_cast<float*>(filterKernel.data()), &status);
	CHECK_ERROR(status, "clCreateBuffer");

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterWeightsBuffer);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputImage);
}

void Convolution3DCL::execute()
{
	status = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, size, nullptr,
	                                 0, nullptr, nullptr);
	CHECK_ERROR(status, "clEnqueueNDRangeKernel");
}

void Convolution3DCL::getResult(image_stack_ref result)
{
	std::size_t origin [3] = {0, 0, 0};
	std::size_t region [3] = {result.shape()[0],
	                          result.shape()[1],
	                          result.shape()[2]};
	clEnqueueReadImage(queue, outputImage, CL_TRUE,
	                   origin, region, 0, 0,
	                   result.data(), 0, nullptr, nullptr);

	// std::cout << region[0] << " " << region[1] << " " << region[2] << std::endl;
	// for(int i = 0; i < 10; i++){
	// 	for(int j = 0; j < 10; j++){
	// 		for(int k = 0; k < 10; k++){
	// 			std::cout << result.data()[i*100+j*10+k] << " ";
	// 		}std::cout << std::endl;
	// 	}std::cout << std::endl;
	// }std::cout << std::endl;
}

void Convolution3DCL::checkError(cl_int status, const char* label, const char* file, int line)
{
	if(status == CL_SUCCESS)
	{
		return;
	}

	std::cerr << "OpenCL error (in file " << file << " in function " << label << ", line " << line << "): ";
	switch(status)
	{
	case CL_BUILD_PROGRAM_FAILURE:
		std::cerr << "CL_BUILD_PROGRAM_FAILURE" << std::endl;
		break;
	case CL_COMPILER_NOT_AVAILABLE:
		std::cerr << "CL_COMPILER_NOT_AVAILABLE" << std::endl;
		break;
	case CL_DEVICE_NOT_AVAILABLE:
		std::cerr << "CL_DEVICE_NOT_AVAILABLE" << std::endl;
		break;
	case CL_DEVICE_NOT_FOUND:
		std::cerr << "CL_DEVICE_NOT_FOUND" << std::endl;
		break;
	case CL_IMAGE_FORMAT_MISMATCH:
		std::cerr << "CL_IMAGE_FORMAT_MISMATCH" << std::endl;
		break;
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		std::cerr << "CL_IMAGE_FORMAT_NOT_SUPPORTED" << std::endl;
		break;
	case CL_INVALID_ARG_INDEX:
		std::cerr << "CL_INVALID_ARG_INDEX" << std::endl;
		break;
	case CL_INVALID_ARG_SIZE:
		std::cerr << "CL_INVALID_ARG_SIZE" << std::endl;
		break;
	case CL_INVALID_ARG_VALUE:
		std::cerr << "CL_INVALID_ARG_VALUE" << std::endl;
		break;
	case CL_INVALID_BINARY:
		std::cerr << "CL_INVALID_BINARY" << std::endl;
		break;
	case CL_INVALID_BUFFER_SIZE:
		std::cerr << "CL_INVALID_BUFFER_SIZE" << std::endl;
		break;
	case CL_INVALID_BUILD_OPTIONS:
		std::cerr << "CL_INVALID_BUILD_OPTIONS" << std::endl;
		break;
	case CL_INVALID_COMMAND_QUEUE:
		std::cerr << "CL_INVALID_COMMAND_QUEUE" << std::endl;
		break;
	case CL_INVALID_CONTEXT:
		std::cerr << "CL_INVALID_CONTEXT" << std::endl;
		break;
	case CL_INVALID_DEVICE:
		std::cerr << "CL_INVALID_DEVICE" << std::endl;
		break;
	case CL_INVALID_DEVICE_TYPE:
		std::cerr << "CL_INVALID_DEVICE_TYPE" << std::endl;
		break;
	case CL_INVALID_EVENT:
		std::cerr << "CL_INVALID_EVENT" << std::endl;
		break;
	case CL_INVALID_EVENT_WAIT_LIST:
		std::cerr << "CL_INVALID_EVENT_WAIT_LIST" << std::endl;
		break;
	case CL_INVALID_GL_OBJECT:
		std::cerr << "CL_INVALID_GL_OBJECT" << std::endl;
		break;
	case CL_INVALID_GLOBAL_OFFSET:
		std::cerr << "CL_INVALID_GLOBAL_OFFSET" << std::endl;
		break;
	case CL_INVALID_HOST_PTR:
		std::cerr << "CL_INVALID_HOST_PTR" << std::endl;
		break;
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		std::cerr << "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR" << std::endl;
		break;
	case CL_INVALID_IMAGE_SIZE:
		std::cerr << "CL_INVALID_IMAGE_SIZE" << std::endl;
		break;
	case CL_INVALID_KERNEL_NAME:
		std::cerr << "CL_INVALID_KERNEL_NAME" << std::endl;
		break;
	case CL_INVALID_KERNEL:
		std::cerr << "CL_INVALID_KERNEL" << std::endl;
		break;
	case CL_INVALID_KERNEL_ARGS:
		std::cerr << "CL_INVALID_KERNEL_ARGS" << std::endl;
		break;
	case CL_INVALID_KERNEL_DEFINITION:
		std::cerr << "CL_INVALID_KERNEL_DEFINITION" << std::endl;
		break;
	case CL_INVALID_MEM_OBJECT:
		std::cerr << "CL_INVALID_MEM_OBJECT" << std::endl;
		break;
	case CL_INVALID_OPERATION:
		std::cerr << "CL_INVALID_OPERATION" << std::endl;
		break;
	case CL_INVALID_PLATFORM:
		std::cerr << "CL_INVALID_PLATFORM" << std::endl;
		break;
	case CL_INVALID_PROGRAM:
		std::cerr << "CL_INVALID_PROGRAM" << std::endl;
		break;
	case CL_INVALID_PROGRAM_EXECUTABLE:
		std::cerr << "CL_INVALID_PROGRAM_EXECUTABLE" << std::endl;
		break;
	case CL_INVALID_QUEUE_PROPERTIES:
		std::cerr << "CL_INVALID_QUEUE_PROPERTIES" << std::endl;
		break;
	case CL_INVALID_SAMPLER:
		std::cerr << "CL_INVALID_SAMPLER" << std::endl;
		break;
	case CL_INVALID_VALUE:
		std::cerr << "CL_INVALID_VALUE" << std::endl;
		break;
	case CL_INVALID_WORK_DIMENSION:
		std::cerr << "CL_INVALID_WORK_DIMENSION" << std::endl;
		break;
	case CL_INVALID_WORK_GROUP_SIZE:
		std::cerr << "CL_INVALID_WORK_GROUP_SIZE" << std::endl;
		break;
	case CL_INVALID_WORK_ITEM_SIZE:
		std::cerr << "CL_INVALID_WORK_ITEM_SIZE" << std::endl;
		break;
	case CL_MAP_FAILURE:
		std::cerr << "CL_MAP_FAILURE" << std::endl;
		break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		std::cerr << "CL_MEM_OBJECT_ALLOCATION_FAILURE" << std::endl;
		break;
	case CL_MEM_COPY_OVERLAP:
		std::cerr << "CL_MEM_COPY_OVERLAP" << std::endl;
		break;
	case CL_OUT_OF_HOST_MEMORY:
		std::cerr << "CL_OUT_OF_HOST_MEMORY" << std::endl;
		break;
	case CL_OUT_OF_RESOURCES:
		std::cerr << "CL_OUT_OF_RESOURCES" << std::endl;
		break;
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		std::cerr << "CL_PROFILING_INFO_NOT_AVAILABLE" << std::endl;
		break;
	}
	exit(status);
}

} /* namespace opencl */
} /* namespace anyfold */
