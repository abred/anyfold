#ifndef _TEST_FIXTURES_H_
#define _TEST_FIXTURES_H_
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

template <unsigned short KernelDimSize = 3, 
	  unsigned ImageDimSize = 8
	  >
struct convolutionFixture3D
{
	const unsigned    image_size_;
	std::vector<int>  image_shape_;
	std::vector<int>  padded_image_shape_;

	image_stack       image_;
	image_stack       padded_image_;
	image_stack       padded_one_;
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

  
	BOOST_STATIC_ASSERT(KernelDimSize % 2 != 0);

public:
  
	convolutionFixture3D():
		image_size_((unsigned)std::pow(ImageDimSize,3)),
		image_shape_(3,ImageDimSize),
		padded_image_shape_(3,ImageDimSize+2*(KernelDimSize/2)),

		image_        (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
		padded_image_ (boost::extents[ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)]),
		padded_one_   (boost::extents[ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)]),
		output_       (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
		padded_output_(boost::extents[ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)]),

		image_folded_by_horizontal_(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
		image_folded_by_vertical_  (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
		image_folded_by_depth_     (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
		image_folded_by_all1_      (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
		padded_image_folded_by_horizontal_(boost::extents[ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)]),
		padded_image_folded_by_vertical_  (boost::extents[ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)]),
		padded_image_folded_by_depth_     (boost::extents[ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)]),
		padded_image_folded_by_all1_      (boost::extents[ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)]),

		kernel_size_((unsigned)std::pow(KernelDimSize,3)),
		kernel_dims_(3,KernelDimSize),
		asymm_kernel_dims_(3,KernelDimSize),
			
		trivial_kernel_   (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
		identity_kernel_  (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
		vertical_kernel_  (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
		horizontal_kernel_(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
		depth_kernel_     (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
		all1_kernel_      (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize])
	{
    
		//FILL KERNELS
		const unsigned halfKernel  = KernelDimSize/2u;
        
		std::fill(trivial_kernel_.data(),    trivial_kernel_.data()+ kernel_size_,    0.f);
		std::fill(identity_kernel_.data(),   identity_kernel_.data()+ kernel_size_,   0.f);
		std::fill(vertical_kernel_.data(),   vertical_kernel_.data()+ kernel_size_,   0.f);
		std::fill(depth_kernel_.data(),      depth_kernel_.data()+ kernel_size_,      0.f);
		std::fill(all1_kernel_.data(),       all1_kernel_.data()+ kernel_size_,       1.f);
		std::fill(horizontal_kernel_.data(), horizontal_kernel_.data()+ kernel_size_, 0.f);


		identity_kernel_[KernelDimSize/2][KernelDimSize/2][KernelDimSize/2] = 1.; 

		for(unsigned int index = 0;index<KernelDimSize;++index){
			horizontal_kernel_[index][halfKernel][halfKernel] = float(index+1);
			vertical_kernel_[halfKernel][index][halfKernel] = float(index+1);
			depth_kernel_   [halfKernel][halfKernel][index] = float(index+1);
		}
    
		//FILL IMAGES
		unsigned image_axis = ImageDimSize;
		unsigned image_size = std::pow(image_axis,3);
		unsigned padded_image_axis = ImageDimSize+2*halfKernel;
		unsigned padded_image_size = std::pow(padded_image_axis,3);
		
		std::fill(padded_image_.data(),
			  padded_image_.data() + padded_image_size,  0.f);
		std::fill(padded_one_.data(),
			  padded_one_.data() + padded_image_size,    0.f);
		std::fill(output_.data(),
			  output_.data() + image_size,               0.f);
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

		padded_one_[padded_image_axis/2][padded_image_axis/2][padded_image_axis/2] = 1.f;

		for (unsigned pixel = 0; pixel < image_.num_elements(); ++pixel)
		{
			image_.data()[pixel] = float(pixel);
		}


		//PADD THE IMAGE FOR CONVOLUTION
		range axis_subrange = range(halfKernel,halfKernel+ImageDimSize);
		image_stack_view padded_image_original = padded_image_[ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
		padded_image_original = image_;
    
		padded_image_folded_by_horizontal_  = padded_image_;
		padded_image_folded_by_vertical_    = padded_image_;
		padded_image_folded_by_depth_       = padded_image_;
		padded_image_folded_by_all1_        = padded_image_;

		//PREPARE ASYMM IMAGES
		std::vector<unsigned> symm_offsets(3);
		std::fill(symm_offsets.begin(), symm_offsets.end(), halfKernel);

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
		image_folded_by_horizontal_  = padded_image_folded_by_horizontal_[ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
		image_folded_by_vertical_    = padded_image_folded_by_vertical_[ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
		image_folded_by_depth_       = padded_image_folded_by_depth_[ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
		image_folded_by_all1_        = padded_image_folded_by_all1_[ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];

	}
  
	virtual ~convolutionFixture3D()  { 
    
	};
    
	static const unsigned image_axis_size = ImageDimSize;
	static const unsigned kernel_axis_size = KernelDimSize;

};

typedef convolutionFixture3D<> default_3D_fixture;


}

#endif
