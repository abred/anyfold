#ifndef _GPU_CONVOLVE_HPP_
#define _GPU_CONVOLVE_HPP_

#include <vector>

#include "image_stack_utils.h"
#include "convolution3DCL.hpp"

namespace anyfold {

namespace gpu {
    
void convolve(image_stack_cref image, 
              image_stack_cref kernel, 
              image_stack_ref result,
              const std::vector<int>& offset)
{
	Convolution3DCL c;
	c.setupCLcontext();
	c.createProgramAndLoadKernel("convolution3d.cl", "convolution3d");
	c.setupKernelArgs(image, kernel, offset);
	c.execute();
	c.getResult(result);
}


void convolve_3d(const float* src_begin, int* src_extents,
                 float* kernel_begin, int* kernel_extents,
                 float* out_begin)
{
	std::vector<int> image_shape(src_extents,src_extents+3);
	std::vector<int> kernel_shape(kernel_extents,kernel_extents+3);
      
	anyfold::image_stack_cref image(src_begin, image_shape);
	anyfold::image_stack_cref kernel(kernel_begin, kernel_shape);
	anyfold::image_stack_ref output(out_begin, image_shape);

	std::vector<int> offsets(3);
	for (unsigned i = 0; i < offsets.size(); ++i)
		offsets[i] = kernel_shape[i]/2;
      
	convolve(image,kernel,output,offsets);
}

} /* namespace gpu */
} /* namespace anyfold */

#endif /* _GPU_CONVOLVE_H_ */
