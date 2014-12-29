#include "gpu/convolution3DCL.hpp"

#include <iostream>
#include <fstream>

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

void Convolution3DCL::createProgramAndLoadKernel(const char* fileName, const char* kernelName)
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

	createProgram(content);
	loadKernel(kernelName);
}

void Convolution3DCL::createProgram(const std::string& source)
{
	const char* sources [1] = { source.c_str() };

	program = clCreateProgramWithSource(context, 1, sources, nullptr, &status);
	CHECK_ERROR(status, "clCreateProgramWithSource");

	status = clBuildProgram(program, 0, nullptr,
	                        "-D FILTER_SIZE=3 -D FILTER_SIZE_HALF=1",
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

	if (platformCount == 0)
	{
		std::cerr << "No OpenCL platform found" << std::endl;
		return false;
	}
	else
	{
		std::cout << "Found " << platformCount << " platform(s)" << std::endl;
	}

	platforms.resize(platformCount);
	clGetPlatformIDs(platformCount, platforms.data(), nullptr);

	for (cl_uint i = 0; i < platformCount; ++i)
	{
		std::cout << "\t (" << (i+1) << ") : "
		          << getPlatformInfo(platforms[i], CL_PLATFORM_NAME)
		          << "\n\tExtensions: \n"
		          << "\t" << getPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS)
		          << std::endl;
	}
	std::cout << "Platform (1) chosen" << std::endl;

	cl_uint deviceCount = 0;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, nullptr,
	                         &deviceCount);
	CHECK_ERROR(status, "clGetDeviceIDs");

	if (deviceCount == 0)
	{
		std::cerr << "No OpenCL device found" << std::endl;
		return false;
	}
	else
	{
		std::cout << "Found " << deviceCount << " device(s)" << std::endl;
	}

	devices.resize(deviceCount);
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, deviceCount,
	               devices.data(), nullptr);
	CHECK_ERROR(status, "clGetDeviceIDs");

	for (cl_uint i = 0; i < deviceCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << getDeviceName(devices[i])
		          << "\n\tExtensions: \n";
		char* str = (char*)getDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS);
		std::cout << "\t" << str << std::endl << std::endl;
		free(str);
	}
	std::cout << "Device (1) chosen" << std::endl;

	const cl_context_properties contextProperties [] =
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties> (platforms[0]),
		0
	};

	context = clCreateContext(contextProperties, deviceCount,
	                                     devices.data(), errorCallback, nullptr, &status);
	CHECK_ERROR(status, "clCreateContext");

	std::cout << "Context created" << std::endl;
	return true;
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
