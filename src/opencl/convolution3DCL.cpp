#include <iostream>
#include <fstream>

#include "opencl/convolution3DCL.hpp"

namespace anyfold {

namespace opencl {

#define CHECK_ERROR(status, fctname) {					\
		checkError(status, fctname, __FILE__, __LINE__ -1);	\
	}

void CL_CALLBACK errorCallback(const char *errinfo,
                               const void *private_info, size_t cb,
                               void *user_data)
{
    std::cerr << errinfo << " (reported by error callback)" << std::endl;
}

std::string Convolution3DCL::getPlatformInfo(cl::Platform platform, cl_platform_info info)
{
    std::string result;
    cl_int status = platform.getInfo(info,&result);
    CHECK_ERROR(status, "cl::Platform::getInfo");
    return result;
}

std::string Convolution3DCL::getDeviceName(cl::Device device)
{
    std::string result;
    cl_int status = device.getInfo(CL_DEVICE_NAME,&result);
    CHECK_ERROR(status, "cl::Device::getInfo");
    return result;
}

std::string Convolution3DCL::getDeviceInfo(cl::Device device, cl_device_info info)
{
    std::string result;
    status = device.getInfo(info,&result);
    CHECK_ERROR(status, "cl::Device::getInfo");
    return result;
}


void Convolution3DCL::createProgramAndLoadKernel(const std::string& fileName, const std::string& kernelName, image_stack_cref _image,image_stack_cref _kernel)
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

    createProgram(content, _image, _kernel);
    loadKernel(kernelName);
}

void Convolution3DCL::createProgram(const std::string& source,
                                    image_stack_cref _image,
                                    image_stack_cref _kernel)
{
    kernel_shape[0] = _kernel.shape()[0];
    kernel_shape[1] = _kernel.shape()[1];
    kernel_shape[2] = _kernel.shape()[2];

    image_shape[0] = _image.shape()[0];
    image_shape[1] = _image.shape()[1];
    image_shape[2] = _image.shape()[2];

    cl::Program::Sources program_source(1, std::make_pair(source.c_str(), source.length()));

    program = cl::Program(context, program_source, &status);
    CHECK_ERROR(status, "cl::Program");

    std::string defines = std::string(" -D FILTER_SIZE=") + std::to_string(kernel_shape[0]) +
                          std::string(" -D FILTER_SIZE_X=") + std::to_string(kernel_shape[0]) +
                          std::string(" -D FILTER_SIZE_Y=") + std::to_string(kernel_shape[1]) +
                          std::string(" -D FILTER_SIZE_Z=") + std::to_string(kernel_shape[2]) +
                          std::string(" -D FILTER_SIZE_HALF=") + std::to_string(kernel_shape[0]/2) +
                          std::string(" -D FILTER_SIZE_HALF_X=") + std::to_string(kernel_shape[0]/2) +
                          std::string(" -D FILTER_SIZE_HALF_Y=") + std::to_string(kernel_shape[1]/2) +
                          std::string(" -D FILTER_SIZE_HALF_Z=") + std::to_string(kernel_shape[2]/2) +
                          std::string(" -D IMAGE_SIZE_X=") + std::to_string(image_shape[0]) +
			  std::string(" -D IMAGE_SIZE_Y=") + std::to_string(image_shape[1]) +
			  std::string(" -D IMAGE_SIZE_Z=") + std::to_string(image_shape[2]);

    status = program.build(devices,
                           defines.c_str(),
                           nullptr, nullptr);

    std::string log;
    program.getBuildInfo(devices[0],CL_PROGRAM_BUILD_LOG,&log);
    if(log.size() > 0)
    {
        std::cout << log << std::endl;
    }
    CHECK_ERROR(status, "cl::Program::Build");
}

void Convolution3DCL::loadKernel(const std::string& kernelName)
{
    kernel = cl::Kernel(program,kernelName.c_str(), &status);
    CHECK_ERROR(status, "cl::Kernel");
}

bool Convolution3DCL::setupCLcontext()
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

void Convolution3DCL::setupKernelArgs(image_stack_cref image,
                                      image_stack_cref filterKernel,
                                      const std::vector<int>& offset)
{
    inputBuffer = cl::Buffer(context,
                             CL_MEM_READ_ONLY |
                             CL_MEM_COPY_HOST_PTR,
                             sizeof(float) * image.num_elements(),
                             const_cast<float*>(image.data()), &status);
    CHECK_ERROR(status, "cl::Buffer");

    outputBuffer = cl::Buffer(context,
                              CL_MEM_WRITE_ONLY,
                              sizeof(float) * image.num_elements(),
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
    kernel.setArg(2,outputBuffer);
}

void Convolution3DCL::execute()
{
    queue.enqueueNDRangeKernel(kernel,0,cl::NDRange(image_shape[0],image_shape[1],image_shape[2]));
    CHECK_ERROR(status, "Queue::enqueueNDRangeKernel");
}

void Convolution3DCL::getResult(image_stack_ref result)
{
    status = queue.enqueueReadBuffer(outputBuffer, CL_TRUE,
                                     0, sizeof(float) * result.num_elements(),
                                     result.data());
    CHECK_ERROR(status, "Queue::enqueueReadImage");
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
