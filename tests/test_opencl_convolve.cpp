#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE OPENCL_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include <boost/mpl/vector.hpp>
#include "test_fixtures.hpp"
#include <numeric>
#include <algorithm>
#include "anyfold.hpp"

#include "test_algorithms.hpp"
#include "image_stack_utils.h"

static anyfold::storage local_order = boost::c_storage_order();

typedef anyfold::convolutionFixture3D<3,32> fixture_3D_32_3;
typedef anyfold::convolutionFixture3D<3,64> fixture_3D_64_3;
typedef anyfold::convolutionFixture3D<5,32> fixture_3D_32_5;
typedef anyfold::convolutionFixture3DAsym<3,3,3,12,12,12> fixture_3D_asym_3_3_3;
typedef anyfold::convolutionFixture3DAsym<3,5,7,12,12,12> fixture_3D_asym_3_5_7;

typedef boost::mpl::vector<
	anyfold::default_3D_fixture
// 	, fixture_3D_32_3
// 	, fixture_3D_64_3
// 	, fixture_3D_32_5
	, fixture_3D_asym_3_3_3
	, fixture_3D_asym_3_5_7
	> Fixtures;

BOOST_FIXTURE_TEST_CASE_TEMPLATE(trivial_convolve, T, Fixtures, T)
{
	const float* image = T::padded_image_.data();
	anyfold::image_stack expected(T::image_);


	float* kernel = new float[T::kernel_size_];
	std::fill(kernel, kernel+T::kernel_size_,0.f);

	anyfold::opencl::convolve_3d(image, &T::padded_image_shape_[0],
	                             kernel,&T::kernel_dims_[0],
	                             T::padded_output_.data());

	float sum = std::accumulate(T::padded_output_.data(),
	                            T::padded_output_.data() + T::padded_output_.num_elements(),0.f);

	BOOST_CHECK_CLOSE(sum, 0.f, .00001);

	delete [] kernel;
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(identity_convolve, T, Fixtures, T)
{
	float sum_original = std::accumulate(T::padded_image_.data(),
	                                     T::padded_image_.data() + T::padded_image_.num_elements(),
	                                     0.f);

	anyfold::opencl::convolve_3d(T::padded_image_.data(),(int*)&T::padded_image_shape_[0],
	                             T::identity_kernel_.data(),&T::kernel_dims_[0],
	                             T::padded_output_.data());

	float sum = std::accumulate(T::padded_output_.data(),
	                            T::padded_output_.data() +T::padded_output_.num_elements(),
	                            0.f);
	BOOST_CHECK_CLOSE(sum, sum_original, .00001);

	float l2norm = anyfold::l2norm(T::padded_image_.data(),
				       T::padded_output_.data(),
				       T::padded_output_.num_elements());
	BOOST_CHECK_CLOSE(l2norm, 0, .00001);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(horizontal_convolve, T, Fixtures, T)
{
	anyfold::opencl::convolve_3d(T::padded_image_.data(),(int*)&T::padded_image_shape_[0],
	                             T::horizontal_kernel_.data(),&T::kernel_dims_[0],
	                             T::padded_output_.data());

	float sum = std::accumulate(T::padded_output_.data(),
	                            T::padded_output_.data() + T::padded_output_.num_elements(),
	                            0.f);
	float sum_expected = std::accumulate(T::padded_image_folded_by_horizontal_.data(),
	                                     T::padded_image_folded_by_horizontal_.data() +
					     T::padded_image_folded_by_horizontal_.num_elements(),
	                                     0.f);

	BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001f);
	float l2norm = anyfold::l2norm(T::padded_output_.data(),
				       T::padded_image_folded_by_horizontal_.data(),
				       T::padded_output_.num_elements());
	BOOST_REQUIRE_CLOSE(l2norm, 0, .00001);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(vertical_convolve, T, Fixtures, T)
{
	anyfold::opencl::convolve_3d(T::padded_image_.data(),(int*)&T::padded_image_shape_[0],
	                             T::vertical_kernel_.data(),&T::kernel_dims_[0],
	                             T::padded_output_.data());

	float sum = std::accumulate(T::padded_output_.data(),
	                            T::padded_output_.data() + T::padded_output_.num_elements(),
	                            0.f);
	float sum_expected = std::accumulate(T::padded_image_folded_by_vertical_.data(),
	                                     T::padded_image_folded_by_vertical_.data() +
					     T::padded_image_folded_by_vertical_.num_elements(),
	                                     0.f);

	BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001f);
	float l2norm = anyfold::l2norm(T::padded_output_.data(),
				       T::padded_image_folded_by_vertical_.data(),
				       T::padded_output_.num_elements());
	BOOST_REQUIRE_CLOSE(l2norm, 0, .00001);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(depth_convolve, T, Fixtures, T)
{
	anyfold::opencl::convolve_3d(T::padded_image_.data(),(int*)&T::padded_image_shape_[0],
	                             T::depth_kernel_.data(),&T::kernel_dims_[0],
	                             T::padded_output_.data());

	float sum = std::accumulate(T::padded_output_.data(),
	                            T::padded_output_.data() + T::padded_output_.num_elements(),
	                            0.f);
	float sum_expected = std::accumulate(T::padded_image_folded_by_depth_.data(),
	                                     T::padded_image_folded_by_depth_.data() +
					     T::padded_image_folded_by_depth_.num_elements(),
	                                     0.f);

	BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001f);
	float l2norm = anyfold::l2norm(T::padded_output_.data(),
				       T::padded_image_folded_by_depth_.data(),
				       T::padded_output_.num_elements());
	BOOST_REQUIRE_CLOSE(l2norm, 0, .00001);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(all1_convolve, T, Fixtures, T)
{
	anyfold::opencl::convolve_3d(T::padded_image_.data(),(int*)&T::padded_image_shape_[0],
	                             T::all1_kernel_.data(),&T::kernel_dims_[0],
	                             T::padded_output_.data());

	float sum = std::accumulate(T::padded_output_.data(),
	                            T::padded_output_.data() + T::padded_output_.num_elements(),
	                            0.f);
	float sum_expected = std::accumulate(T::padded_image_folded_by_all1_.data(),
	                                     T::padded_image_folded_by_all1_.data() +
					     T::padded_image_folded_by_all1_.num_elements(),
	                                     0.f);
	std::cout << T::kernel_dims_[0] << "," << T::kernel_dims_[1] << "," << T::kernel_dims_[2] << std::endl;
	int glbl = 0;
	for(int z = 0; z < T::padded_output_.shape()[2]; z++) {
	  for(int y = 0; y < T::padded_output_.shape()[1]; y++) {
	    for(int x = 0; x < T::padded_output_.shape()[0]; x++) {
	      std::cout << x << "," << y << "," << z << ": " << T::padded_output_[x][y][z] << "/" << T::padded_image_folded_by_all1_ [x][y][z] << std::endl;	     
	      std::cout << "G: " << glbl << "/" << 
		    x + T::padded_output_.shape()[0] * (y +  T::padded_output_.shape()[1] * z);
		   glbl++;
		   
		   if (x <  T::kernel_dims_[0]/2 ||
			x > T::padded_output_.shape()[0] -  (T::kernel_dims_[0]/2) - 1 ||
			y < T::kernel_dims_[1]/2 ||
			y > T::padded_output_.shape()[1] - (T::kernel_dims_[1]/2) - 1 ||
			z < T::kernel_dims_[2]/2 ||
			z > T::padded_output_.shape()[2]- (T::kernel_dims_[2]/2) - 1
		) { std::cout << " true"; }
		else { std::cout << " false"; }
		std::cout << std::endl;
	    }
	  }
	}
	
	
	BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001f);
	float l2norm = anyfold::l2norm(T::padded_output_.data(),
				       T::padded_image_folded_by_all1_.data(),
				       T::padded_output_.num_elements());
	BOOST_REQUIRE_CLOSE(l2norm, 0, .00001);
}
