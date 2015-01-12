#ifndef _OPENCL_CONVOLVE_HPP_
#define _OPENCL_CONVOLVE_HPP_

#include "opencl_utils.hpp"
#include "image_stack_utils.h"

namespace anyfold {

namespace opencl {
  
template <typename ImageStackT,typename CImageStackT, typename DimT>
void convolve_simple(CImageStackT& _image,
                     CImageStackT& _kernel,
                     ImageStackT& _result,
                     const std::vector<DimT>& _offset) {
    //create needed opnecl stuff
    cl::Context context = createCLContext(CL_DEVICE_TYPE_ALL,VENDOR_ANY);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue(context,devices[0]);

    //comfort variables
    //size_t image_size = sizeof(_image.element)*_image.size();
    size_t kernel_size = sizeof(float)*_kernel.num_elements();//TODO
    size_t result_size = sizeof(float)*_result.num_elements();
//     size_t result_size = sizeof(float)*(_result.shape()[0] * _result.shape()[1] * _result.shape()[2]);//TODO

    //dimensionality
    unsigned long int image_shape[] = {_image.shape()[0],_image.shape()[1],_image.shape()[2]}; //TODO
    unsigned long int kernel_shape[] = {_kernel.shape()[0],_kernel.shape()[1],_kernel.shape()[2]};//TODO
    //size_t* result_shape = _result.shape();

    //get compiled kernel
    cl::Program program = buildProgramFromSource(context,"../../include/opencl/convolve_kernel_simple.cl",""); //TODO always correct path

    //flatten image into one array //TODO
    float* image_data = const_cast<float*>(_image.data());
    float* kernel_data = const_cast<float*>(_kernel.data());
    float* result_data = const_cast<float*>(_result.data());
    
    //TODO half the kernel extends
    for(int i = 0; i < 3; i++) {
	kernel_shape[i] = (unsigned long int)(kernel_shape[i]/2.);
    }

    //allocate memory on device
    // Create an OpenCL Image / texture and transfer data to the device
    cl::Image3D cl_image = cl::Image3D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_FLOAT),
                                       image_shape[0], image_shape[1],image_shape[2], 0, 0, image_data);
    
    cl::Buffer cl_kernel = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kernel_size, kernel_data);

    cl::Buffer cl_kernel_shape = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned long int)*3, &kernel_shape);

    cl::Buffer cl_result = cl::Buffer(context, CL_MEM_WRITE_ONLY, result_size);

    //create kernel
    cl::Kernel kernel(program,"convolve_simple");

    // Set arguments to kernel
    kernel.setArg(0, cl_image);
    kernel.setArg(1, cl_kernel_shape);
    kernel.setArg(2, cl_kernel);
    kernel.setArg(3, cl_result);

    // Run the kernel on every pixel/voxel of image
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_shape[0],image_shape[1],image_shape[2]),cl::NullRange);

    //writeback to host
    queue.enqueueReadBuffer(cl_result, CL_TRUE, 0, result_size, result_data);
    
//     std::cout << "check 1: "<< result_data[0];
//     std::cout << ";" << _result[0][0][0] << std::endl;
    
    
}
  

template <typename ImageStackT,typename CImageStackT, typename DimT>
void convolve(CImageStackT& _image,
              CImageStackT& _kernel,
              ImageStackT& _result,
              const std::vector<DimT>& _offset) {
    convolve_simple(_image,_kernel,_result,_offset);
}


template <typename ExtentT, typename SrcIterT, typename KernIterT, typename OutIterT>
void convolve_3d(SrcIterT src_begin, ExtentT* src_extents,
                 KernIterT kernel_begin, ExtentT* kernel_extents,
                 OutIterT out_begin)
{
    std::vector<ExtentT> image_shape(src_extents,src_extents+3);
    std::vector<ExtentT> kernel_shape(kernel_extents,kernel_extents+3);

    anyfold::image_stack_cref image(src_begin, image_shape);
    anyfold::image_stack_cref kernel(kernel_begin, kernel_shape);
    anyfold::image_stack_ref output(out_begin, image_shape);

    std::vector<ExtentT> offsets(3);
    for (unsigned i = 0; i < offsets.size(); ++i)
        offsets[i] = kernel_shape[i]/2;

    return convolve(image,kernel,output,offsets);
}

template <typename ExtentT, typename SrcIterT, typename KernIterT, typename OutIterT>
void discrete_convolve_3d(SrcIterT src_begin, ExtentT* src_extents,
                          KernIterT kernel_begin, ExtentT* kernel_extents,
                          OutIterT out_begin) {

}


};
};

#endif /* _OPENCL_CONVOLVE_HPP_ */
