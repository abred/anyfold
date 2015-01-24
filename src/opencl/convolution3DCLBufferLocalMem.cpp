#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "opencl/convolution3DCLBufferLocalMem.hpp"

namespace anyfold {

namespace opencl {

#define CHECK_ERROR(status, fctname) {					\
		checkError(status, fctname, __FILE__, __LINE__ -1);	\
	}

static void CL_CALLBACK errorCallback(const char *errinfo,
                               const void *private_info, size_t cb,
                               void *user_data)
{
	std::cerr << errinfo << " (reported by error callback)" << std::endl;
}

std::string Convolution3DCLBufferLocalMem::getPlatformInfo(cl::Platform platform, cl_platform_info info)
{
	std::string result;
	cl_int status = platform.getInfo(info,&result);
	CHECK_ERROR(status, "cl::Platform::getInfo");
	return result;
}

std::string Convolution3DCLBufferLocalMem::getDeviceName(cl::Device device)
{
	std::string result;
	cl_int status = device.getInfo(CL_DEVICE_NAME,&result);
	CHECK_ERROR(status, "cl::Device::getInfo");
	return result;
}

std::string Convolution3DCLBufferLocalMem::getDeviceInfo(cl::Device device, cl_device_info info)
{
	std::string result;
	status = device.getInfo(info,&result);
	CHECK_ERROR(status, "cl::Device::getInfo");
	return result;
}


void Convolution3DCLBufferLocalMem::createProgramAndLoadKernel(const std::string& fileName, const std::string& kernelName, size_t const* filterSize)
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

void Convolution3DCLBufferLocalMem::createProgram(const std::string& source, 
                                    size_t const* fs)
{
	cl::Program::Sources program_source(1, std::make_pair(source.c_str(), source.length()));

	program = cl::Program(context, program_source, &status);
	CHECK_ERROR(status, "cl::Program");
	
	std::string defines = std::string("-D FILTER_SIZE_X=") +
	                      std::to_string(fs[2]) +
	                      std::string(" -D FILTER_SIZE_Y=") +
	                      std::to_string(fs[1]) +
	                      std::string(" -D FILTER_SIZE_Z=") +
	                      std::to_string(fs[0]) +
	                      std::string(" -D FILTER_SIZE_X_HALF=") +
	                      std::to_string(fs[2]/2) +
	                      std::string(" -D FILTER_SIZE_Y_HALF=") +
	                      std::to_string(fs[1]/2) +
	                      std::string(" -D FILTER_SIZE_Z_HALF=") +
	                      std::to_string(fs[0]/2);
	status = program.build(devices,
	                       defines.c_str(),
	                       nullptr, nullptr);

	if(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) == CL_BUILD_ERROR)
	{
		std::string log;
		program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
		std::cout << log << std::endl;
	}
	CHECK_ERROR(status, "cl::Program::Build");
}

void Convolution3DCLBufferLocalMem::loadKernel(const std::string& kernelName)
{
	kernel = cl::Kernel(program,kernelName.c_str(), &status);
	CHECK_ERROR(status, "cl::Kernel");
}

bool Convolution3DCLBufferLocalMem::setupCLcontext()
{
	status = cl::Platform::get(&platforms);

	status = platforms[0].getDevices(CL_DEVICE_TYPE_ALL,&devices);
	CHECK_ERROR(status, "cl::Platform::getDevices");

	context = cl::Context(devices,nullptr,nullptr,nullptr,&status);
	CHECK_ERROR(status, "cl::Context");

	queue = cl::CommandQueue(context,devices[0],0,&status);
	CHECK_ERROR(status, "cl::CommandQueue");

	return true;
}

void Convolution3DCLBufferLocalMem::setupKernelArgs(image_stack_cref image,
                                      image_stack_cref filterKernel,
                                      const std::vector<int>& offset)
{
	imageSize[0] = image.shape()[2];
	imageSize[1] = image.shape()[1];
	imageSize[2] = image.shape()[0];
	filterSize[0] = filterKernel.shape()[2];
	filterSize[1] = filterKernel.shape()[1];
	filterSize[2] = filterKernel.shape()[0];
	imageSizeInner[0] = imageSize[0]-2*(filterSize[0]/2);
	imageSizeInner[1] = imageSize[1]-2*(filterSize[1]/2);
	imageSizeInner[2] = imageSize[2]-2*(filterSize[2]/2);

	inputBuffer = cl::Buffer(context,
	                         CL_MEM_READ_ONLY |
	                         CL_MEM_COPY_HOST_PTR,
	                         sizeof(float) * image.num_elements(),
	                         const_cast<float*>(image.data()), &status);
	CHECK_ERROR(status, "cl::Buffer");

	size_t imageSizeInnerTotal = imageSizeInner[0] * imageSizeInner[1] * imageSizeInner[2];
	outputBuffer[0] = cl::Buffer(context,
	                             CL_MEM_WRITE_ONLY,
	                             sizeof(float) * imageSizeInnerTotal,
	                             nullptr, &status);
	CHECK_ERROR(status, "cl::Buffer");

	cl_float val = 0.0f;
	queue.enqueueFillBuffer(outputBuffer[0], val, 0, sizeof(float) * imageSizeInnerTotal);

	outputBuffer[1] = cl::Buffer(context,
	                             CL_MEM_WRITE_ONLY,
	                             sizeof(float) * imageSizeInnerTotal,
	                             nullptr, &status);
	CHECK_ERROR(status, "cl::Buffer");
	
	filterWeightsBuffer = cl::Buffer(context,
	                                 CL_MEM_READ_ONLY |
	                                 CL_MEM_COPY_HOST_PTR,
	                                 sizeof(float) * filterKernel.num_elements(),
	                                 const_cast<float*>(filterKernel.data()), &status);
	CHECK_ERROR(status, "cl::Buffer");

	kernel.setArg(0,inputBuffer);
	kernel.setArg(1,filterWeightsBuffer);
	kernel.setArg(2,outputBuffer[0]);
}

void Convolution3DCLBufferLocalMem::execute()
{
	bool d = 0;
	for(int z = 0; z < filterSize[2]; z +=3)
	{
		for(int y = 0; y < filterSize[1]; y +=3)
		{
			for(int x = 0; x < filterSize[0]; x +=3)
			{
				cl_int3 offset = {x, y, z};
				kernel.setArg(4, offset);
				kernel.setArg(2, outputBuffer[d]);
				kernel.setArg(3, outputBuffer[!d]);
				d = !d;
				queue.enqueueNDRangeKernel(kernel, 0,
				                           cl::NDRange(imageSizeInner[0],
				                                       imageSizeInner[1],
				                                       imageSizeInner[2]),
				                           cl::NDRange(4, 4, 4));
				CHECK_ERROR(status, "Queue::enqueueNDRangeKernel");
			}
		}
	}
	outputSwap = d;

	CHECK_ERROR(status, "Queue::enqueueNDRangeKernel");
}

void Convolution3DCLBufferLocalMem::getResult(image_stack_ref result)
{
	cl::size_t<3> bufOffset;
	bufOffset[0] = 0;
	bufOffset[1] = 0;
	bufOffset[2] = 0;
	cl::size_t<3> hostOffset;
	hostOffset[0] = (filterSize[0]/2)*sizeof(float);
	hostOffset[1] = filterSize[1]/2;
	hostOffset[2] = filterSize[2]/2;
	cl::size_t<3> region;
	region[0] = imageSizeInner[0]*sizeof(float);
	region[1] = imageSizeInner[1];
	region[2] = imageSizeInner[2];

	status = queue.enqueueReadBufferRect(outputBuffer[outputSwap], CL_TRUE,
	                                     bufOffset,
	                                     hostOffset,
	                                     region,
	                                     imageSizeInner[0] * sizeof(float),
	                                     imageSizeInner[0] * imageSizeInner[1] * sizeof(float),
	                                     imageSize[0] * sizeof(float),
	                                     imageSize[0] * imageSize[1] * sizeof(float),
	                                     result.data());
	CHECK_ERROR(status, "Queue::enqueueReadBufferRect");
}

void Convolution3DCLBufferLocalMem::checkError(cl_int status, const char* label, const char* file, int line)
{
	if(status == CL_SUCCESS)
	{
		return;
	}

	std::stringstream sstr;
	sstr << "OpenCL error (in file " << file << " in function " << label << ", line " << line << "): ";
	switch(status)
	{
	case CL_BUILD_PROGRAM_FAILURE:
		sstr << "CL_BUILD_PROGRAM_FAILURE" << std::endl;
		break;
	case CL_COMPILER_NOT_AVAILABLE:
		sstr << "CL_COMPILER_NOT_AVAILABLE" << std::endl;
		break;
	case CL_DEVICE_NOT_AVAILABLE:
		sstr << "CL_DEVICE_NOT_AVAILABLE" << std::endl;
		break;
	case CL_DEVICE_NOT_FOUND:
		sstr << "CL_DEVICE_NOT_FOUND" << std::endl;
		break;
	case CL_IMAGE_FORMAT_MISMATCH:
		sstr << "CL_IMAGE_FORMAT_MISMATCH" << std::endl;
		break;
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		sstr << "CL_IMAGE_FORMAT_NOT_SUPPORTED" << std::endl;
		break;
	case CL_INVALID_ARG_INDEX:
		sstr << "CL_INVALID_ARG_INDEX" << std::endl;
		break;
	case CL_INVALID_ARG_SIZE:
		sstr << "CL_INVALID_ARG_SIZE" << std::endl;
		break;
	case CL_INVALID_ARG_VALUE:
		sstr << "CL_INVALID_ARG_VALUE" << std::endl;
		break;
	case CL_INVALID_BINARY:
		sstr << "CL_INVALID_BINARY" << std::endl;
		break;
	case CL_INVALID_BUFFER_SIZE:
		sstr << "CL_INVALID_BUFFER_SIZE" << std::endl;
		break;
	case CL_INVALID_BUILD_OPTIONS:
		sstr << "CL_INVALID_BUILD_OPTIONS" << std::endl;
		break;
	case CL_INVALID_COMMAND_QUEUE:
		sstr << "CL_INVALID_COMMAND_QUEUE" << std::endl;
		break;
	case CL_INVALID_CONTEXT:
		sstr << "CL_INVALID_CONTEXT" << std::endl;
		break;
	case CL_INVALID_DEVICE:
		sstr << "CL_INVALID_DEVICE" << std::endl;
		break;
	case CL_INVALID_DEVICE_TYPE:
		sstr << "CL_INVALID_DEVICE_TYPE" << std::endl;
		break;
	case CL_INVALID_EVENT:
		sstr << "CL_INVALID_EVENT" << std::endl;
		break;
	case CL_INVALID_EVENT_WAIT_LIST:
		sstr << "CL_INVALID_EVENT_WAIT_LIST" << std::endl;
		break;
	case CL_INVALID_GL_OBJECT:
		sstr << "CL_INVALID_GL_OBJECT" << std::endl;
		break;
	case CL_INVALID_GLOBAL_OFFSET:
		sstr << "CL_INVALID_GLOBAL_OFFSET" << std::endl;
		break;
	case CL_INVALID_HOST_PTR:
		sstr << "CL_INVALID_HOST_PTR" << std::endl;
		break;
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		sstr << "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR" << std::endl;
		break;
	case CL_INVALID_IMAGE_SIZE:
		sstr << "CL_INVALID_IMAGE_SIZE" << std::endl;
		break;
	case CL_INVALID_KERNEL_NAME:
		sstr << "CL_INVALID_KERNEL_NAME" << std::endl;
		break;
	case CL_INVALID_KERNEL:
		sstr << "CL_INVALID_KERNEL" << std::endl;
		break;
	case CL_INVALID_KERNEL_ARGS:
		sstr << "CL_INVALID_KERNEL_ARGS" << std::endl;
		break;
	case CL_INVALID_KERNEL_DEFINITION:
		sstr << "CL_INVALID_KERNEL_DEFINITION" << std::endl;
		break;
	case CL_INVALID_MEM_OBJECT:
		sstr << "CL_INVALID_MEM_OBJECT" << std::endl;
		break;
	case CL_INVALID_OPERATION:
		sstr << "CL_INVALID_OPERATION" << std::endl;
		break;
	case CL_INVALID_PLATFORM:
		sstr << "CL_INVALID_PLATFORM" << std::endl;
		break;
	case CL_INVALID_PROGRAM:
		sstr << "CL_INVALID_PROGRAM" << std::endl;
		break;
	case CL_INVALID_PROGRAM_EXECUTABLE:
		sstr << "CL_INVALID_PROGRAM_EXECUTABLE" << std::endl;
		break;
	case CL_INVALID_QUEUE_PROPERTIES:
		sstr << "CL_INVALID_QUEUE_PROPERTIES" << std::endl;
		break;
	case CL_INVALID_SAMPLER:
		sstr << "CL_INVALID_SAMPLER" << std::endl;
		break;
	case CL_INVALID_VALUE:
		sstr << "CL_INVALID_VALUE" << std::endl;
		break;
	case CL_INVALID_WORK_DIMENSION:
		sstr << "CL_INVALID_WORK_DIMENSION" << std::endl;
		break;
	case CL_INVALID_WORK_GROUP_SIZE:
		sstr << "CL_INVALID_WORK_GROUP_SIZE" << std::endl;
		break;
	case CL_INVALID_WORK_ITEM_SIZE:
		sstr << "CL_INVALID_WORK_ITEM_SIZE" << std::endl;
		break;
	case CL_MAP_FAILURE:
		sstr << "CL_MAP_FAILURE" << std::endl;
		break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		sstr << "CL_MEM_OBJECT_ALLOCATION_FAILURE" << std::endl;
		break;
	case CL_MEM_COPY_OVERLAP:
		sstr << "CL_MEM_COPY_OVERLAP" << std::endl;
		break;
	case CL_OUT_OF_HOST_MEMORY:
		sstr << "CL_OUT_OF_HOST_MEMORY" << std::endl;
		break;
	case CL_OUT_OF_RESOURCES:
		sstr << "CL_OUT_OF_RESOURCES" << std::endl;
		break;
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		sstr << "CL_PROFILING_INFO_NOT_AVAILABLE" << std::endl;
		break;
	}
	throw std::runtime_error(sstr.str());
}

} /* namespace opencl */
} /* namespace anyfold */
