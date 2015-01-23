#ifndef _TEST_FIXTURES_ASYM_H_
#define _TEST_FIXTURES_ASYMH_
#include <iostream> 
#include <iomanip> 
#include <vector>
#include <cmath>
//#include "mxn_indexer.hpp"
#include <boost/static_assert.hpp>
#include "boost/multi_array.hpp"

//http://www.boost.org/doc/libs/1_55_0/libs/multi_array/doc/user.html
//http://stackoverflow.com/questions/2168082/how-to-rewrite-array-from-row-order-to-column-order
#include "image_stack_utils.h"
#include "test_algorithms.hpp"

namespace anyfold {

template <unsigned short KernelDimSizeX = 3,
          unsigned short KernelDimSizeY = 3,
          unsigned short KernelDimSizeZ = 3,
          unsigned ImageDimSizeX = 8,
          unsigned ImageDimSizeY = 8,
	  unsigned ImageDimSizeZ = 8
	  >
struct convolutionFixture3DAsym
{
	const unsigned    image_size_;
	std::vector<int>  image_shape_;
	std::vector<int>  padded_image_shape_;

	image_stack       image_;
	image_stack       padded_image_;
	image_stack       output_;
	image_stack       padded_output_;

	image_stack       image_folded_by_horizontal_;
	image_stack       image_folded_by_vertical_;
	image_stack       image_folded_by_depth_;
	image_stack       image_folded_by_all1_;

	image_stack       padded_image_folded_by_horizontal_;
	image_stack       padded_image_folded_by_vertical_;
	image_stack       padded_image_folded_by_depth_;
	image_stack       padded_image_folded_by_all1_;

	const unsigned    kernel_size_;
	std::vector<int>  kernel_dims_;
	std::vector<int>  asymm_kernel_dims_;

	image_stack       trivial_kernel_;
	image_stack       identity_kernel_;
	image_stack       vertical_kernel_;
	image_stack       horizontal_kernel_;
	image_stack       depth_kernel_;
	image_stack       all1_kernel_;

  
	BOOST_STATIC_ASSERT(KernelDimSizeX % 2 != 0);
	BOOST_STATIC_ASSERT(KernelDimSizeY % 2 != 0);
	BOOST_STATIC_ASSERT(KernelDimSizeZ % 2 != 0);

public:
  
	convolutionFixture3DAsym():
		image_size_(ImageDimSizeX * ImageDimSizeY * ImageDimSizeZ),
		image_shape_({ImageDimSizeX, ImageDimSizeY, ImageDimSizeZ}),
		padded_image_shape_ ({ImageDimSizeX+2*(KernelDimSizeX/2),
		                      ImageDimSizeY+2*(KernelDimSizeY/2),
		                      ImageDimSizeZ+2*(KernelDimSizeZ/2)}),

		image_        (boost::extents[ImageDimSizeX][ImageDimSizeY][ImageDimSizeZ]),
		padded_image_ (boost::extents[ImageDimSizeX+2*(KernelDimSizeX/2)][ImageDimSizeY+2*(KernelDimSizeY/2)][ImageDimSizeZ+2*(KernelDimSizeZ/2)]),
		output_       (boost::extents[ImageDimSizeX][ImageDimSizeY][ImageDimSizeZ]),
		padded_output_(boost::extents[ImageDimSizeX+2*(KernelDimSizeX/2)][ImageDimSizeY+2*(KernelDimSizeY/2)][ImageDimSizeZ+2*(KernelDimSizeZ/2)]),

		image_folded_by_horizontal_(boost::extents[ImageDimSizeX][ImageDimSizeY][ImageDimSizeZ]),
		image_folded_by_vertical_  (boost::extents[ImageDimSizeX][ImageDimSizeY][ImageDimSizeZ]),
		image_folded_by_depth_     (boost::extents[ImageDimSizeX][ImageDimSizeY][ImageDimSizeZ]),
		image_folded_by_all1_      (boost::extents[ImageDimSizeX][ImageDimSizeY][ImageDimSizeZ]),
		padded_image_folded_by_horizontal_(boost::extents[ImageDimSizeX+2*(KernelDimSizeX/2)][ImageDimSizeY+2*(KernelDimSizeY/2)][ImageDimSizeZ+2*(KernelDimSizeZ/2)]),
		padded_image_folded_by_vertical_  (boost::extents[ImageDimSizeX+2*(KernelDimSizeX/2)][ImageDimSizeY+2*(KernelDimSizeY/2)][ImageDimSizeZ+2*(KernelDimSizeZ/2)]),
		padded_image_folded_by_depth_     (boost::extents[ImageDimSizeX+2*(KernelDimSizeX/2)][ImageDimSizeY+2*(KernelDimSizeY/2)][ImageDimSizeZ+2*(KernelDimSizeZ/2)]),
		padded_image_folded_by_all1_      (boost::extents[ImageDimSizeX+2*(KernelDimSizeX/2)][ImageDimSizeY+2*(KernelDimSizeY/2)][ImageDimSizeZ+2*(KernelDimSizeZ/2)]),

		kernel_size_(KernelDimSizeX * KernelDimSizeY * KernelDimSizeZ),
		// kernel_size_((unsigned)std::pow(KernelDimSize,3)),
		kernel_dims_ ({KernelDimSizeX, KernelDimSizeY, KernelDimSizeZ}),
		// asymm_kernel_dims_(3,KernelDimSize),
			
		trivial_kernel_   (boost::extents[KernelDimSizeX][KernelDimSizeY][KernelDimSizeZ]),
		identity_kernel_  (boost::extents[KernelDimSizeX][KernelDimSizeY][KernelDimSizeZ]),
		vertical_kernel_  (boost::extents[KernelDimSizeX][KernelDimSizeY][KernelDimSizeZ]),
		horizontal_kernel_(boost::extents[KernelDimSizeX][KernelDimSizeY][KernelDimSizeZ]),
		depth_kernel_     (boost::extents[KernelDimSizeX][KernelDimSizeY][KernelDimSizeZ]),
		all1_kernel_      (boost::extents[KernelDimSizeX][KernelDimSizeY][KernelDimSizeZ])
	{
    
		//FILL KERNELS
		const unsigned halfKernelX  = KernelDimSizeX/2u;
		const unsigned halfKernelY  = KernelDimSizeY/2u;
		const unsigned halfKernelZ  = KernelDimSizeZ/2u;
        
		std::fill(trivial_kernel_.data(),    trivial_kernel_.data()+ kernel_size_,    0.f);
		std::fill(identity_kernel_.data(),   identity_kernel_.data()+ kernel_size_,   0.f);
		std::fill(vertical_kernel_.data(),   vertical_kernel_.data()+ kernel_size_,   0.f);
		std::fill(depth_kernel_.data(),      depth_kernel_.data()+ kernel_size_,      0.f);
		std::fill(all1_kernel_.data(),       all1_kernel_.data()+ kernel_size_,       1.f);
		std::fill(horizontal_kernel_.data(), horizontal_kernel_.data()+ kernel_size_, 0.f);


		identity_kernel_[KernelDimSizeX/2][KernelDimSizeY/2][KernelDimSizeZ/2] = 1.; 

		for(unsigned int index = 0; index < KernelDimSizeX; ++index){
			horizontal_kernel_[index][halfKernelY][halfKernelZ] = float(index+1);
		}
		for(unsigned int index = 0; index < KernelDimSizeY; ++index){
			vertical_kernel_[halfKernelX][index][halfKernelZ] = float(index+1);
		}
		for(unsigned int index = 0; index < KernelDimSizeZ; ++index){
			depth_kernel_   [halfKernelX][halfKernelY][index] = float(index+1);
		}

		//FILL IMAGES
		unsigned padded_image_size = (ImageDimSizeX+2*halfKernelX) *
		                             (ImageDimSizeY+2*halfKernelY) *
		                             (ImageDimSizeZ+2*halfKernelZ);
		
		std::fill(padded_image_.data(),
			  padded_image_.data() + padded_image_size,  0.f);
		std::fill(output_.data(),
			  output_.data() + image_size_,              0.f);
		std::fill(padded_output_.data(),
			  padded_output_.data() + padded_image_size, 0.f);

		std::fill(padded_image_folded_by_horizontal_.data(),
			  padded_image_folded_by_horizontal_.data() + padded_image_size,
			  0.f);
		std::fill(padded_image_folded_by_vertical_.data(),
			  padded_image_folded_by_vertical_.data() + padded_image_size,
			  0.f);
		std::fill(padded_image_folded_by_depth_.data(),
			  padded_image_folded_by_depth_.data() + padded_image_size,
			  0.f);
		std::fill(padded_image_folded_by_all1_.data(),
			  padded_image_folded_by_all1_.data() + padded_image_size,
			  0.f);

		for (unsigned pixel = 0; pixel < image_.num_elements(); ++pixel)
		{
			image_.data()[pixel] = float(pixel);
		}


		//PADD THE IMAGE FOR CONVOLUTION
		range axis_subrangeX = range(halfKernelX,halfKernelX+ImageDimSizeX);
		range axis_subrangeY = range(halfKernelY,halfKernelY+ImageDimSizeY);
		range axis_subrangeZ = range(halfKernelZ,halfKernelZ+ImageDimSizeZ);
		image_stack_view padded_image_original = padded_image_[ boost::indices[axis_subrangeX][axis_subrangeY][axis_subrangeZ] ];
		padded_image_original = image_;
    
		padded_image_folded_by_horizontal_  = padded_image_;
		padded_image_folded_by_vertical_    = padded_image_;
		padded_image_folded_by_depth_       = padded_image_;
		padded_image_folded_by_all1_        = padded_image_;

		//PREPARE ASYMM IMAGES
		std::vector<unsigned> symm_offsets = {halfKernelX, halfKernelY, halfKernelZ};

		//CONVOLVE
		convolve(padded_image_, horizontal_kernel_,
			 padded_image_folded_by_horizontal_, symm_offsets);
		convolve(padded_image_, vertical_kernel_,
			 padded_image_folded_by_vertical_, symm_offsets);
		convolve(padded_image_, depth_kernel_,
			 padded_image_folded_by_depth_, symm_offsets);
		convolve(padded_image_, all1_kernel_,
			 padded_image_folded_by_all1_, symm_offsets);
    
		//EXTRACT NON-PADDED CONTENT FROM CONVOLVED IMAGE STACKS
		image_folded_by_horizontal_  = padded_image_folded_by_horizontal_[ boost::indices[axis_subrangeX][axis_subrangeY][axis_subrangeZ] ];
		image_folded_by_vertical_    = padded_image_folded_by_vertical_[ boost::indices[axis_subrangeX][axis_subrangeY][axis_subrangeZ] ];
		image_folded_by_depth_       = padded_image_folded_by_depth_[ boost::indices[axis_subrangeX][axis_subrangeY][axis_subrangeZ] ];
		image_folded_by_all1_        = padded_image_folded_by_all1_[ boost::indices[axis_subrangeX][axis_subrangeY][axis_subrangeZ] ];

	}
  
	virtual ~convolutionFixture3DAsym() {};
};

typedef convolutionFixture3DAsym<> default_3D_fixture_asym;


}

#endif
