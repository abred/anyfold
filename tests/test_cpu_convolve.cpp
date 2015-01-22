#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE CPU_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <algorithm>
#include "anyfold.hpp"

#include "test_algorithms.hpp"
#include "image_stack_utils.h"

static anyfold::storage local_order = boost::c_storage_order();

BOOST_FIXTURE_TEST_SUITE( convolution_works, anyfold::default_3D_fixture )

BOOST_AUTO_TEST_CASE( trivial_convolve )
{

  const float* image = padded_image_.data();
  anyfold::image_stack expected(image_);


  float* kernel = new float[kernel_size_];
  std::fill(kernel, kernel+kernel_size_,0.f);

  anyfold::cpu::convolve_3d(image, &padded_image_shape_[0],
			    kernel,&kernel_dims_[0],
			    padded_output_.data());

  float sum = std::accumulate(padded_output_.data(),
			      padded_output_.data() + padded_output_.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum, 0.f, .00001);

  delete [] kernel;
}

BOOST_AUTO_TEST_CASE( identity_convolve )
{


  float sum_original = std::accumulate(padded_image_.data(),
				       padded_image_.data() + padded_image_.num_elements(),
				       0.f);

  anyfold::cpu::convolve_3d(padded_image_.data(),(int*)&padded_image_shape_[0],
                            identity_kernel_.data(),&kernel_dims_[0],
                            padded_output_.data());

  float sum = std::accumulate(padded_output_.data(),
			      padded_output_.data() + padded_output_.num_elements(),
			      0.f);
  BOOST_CHECK_CLOSE(sum, sum_original, .00001);

  float l2norm = anyfold::l2norm(padded_image_.data(), padded_output_.data(),  padded_output_.num_elements());
  BOOST_CHECK_CLOSE(l2norm, 0, .00001);
}

BOOST_AUTO_TEST_CASE( horizontal_convolve )
{


  anyfold::cpu::convolve_3d(padded_image_.data(),(int*)&padded_image_shape_[0],
			    horizontal_kernel_.data(),&kernel_dims_[0],
			    padded_output_.data());

  float sum = std::accumulate(padded_output_.data(),
			      padded_output_.data() + padded_output_.num_elements(),
			      0.f);
  float sum_expected = std::accumulate(padded_image_folded_by_horizontal_.data(),
				       padded_image_folded_by_horizontal_.data() + padded_image_folded_by_horizontal_.num_elements(),
				       0.f);

  BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001f);
  float l2norm = anyfold::l2norm(padded_output_.data(), padded_image_folded_by_horizontal_.data(),  padded_output_.num_elements());
  BOOST_REQUIRE_CLOSE(l2norm, 0, .00001);
}

BOOST_AUTO_TEST_CASE( vertical_convolve )
{


  anyfold::cpu::convolve_3d(padded_image_.data(),(int*)&padded_image_shape_[0],
			    vertical_kernel_.data(),&kernel_dims_[0],
			    padded_output_.data());

  float sum = std::accumulate(padded_output_.data(),
			      padded_output_.data() + padded_output_.num_elements(),
			      0.f);
  float sum_expected = std::accumulate(padded_image_folded_by_vertical_.data(),
				       padded_image_folded_by_vertical_.data() + padded_image_folded_by_vertical_.num_elements(),
				       0.f);

  BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001f);
  float l2norm = anyfold::l2norm(padded_output_.data(), padded_image_folded_by_vertical_.data(),  padded_output_.num_elements());
  BOOST_REQUIRE_CLOSE(l2norm, 0, .00001);
}


BOOST_AUTO_TEST_CASE( depth_convolve )
{


  anyfold::cpu::convolve_3d(padded_image_.data(),(int*)&padded_image_shape_[0],
			    depth_kernel_.data(),&kernel_dims_[0],
			    padded_output_.data());

  float sum = std::accumulate(padded_output_.data(),
			      padded_output_.data() + padded_output_.num_elements(),
			      0.f);
  float sum_expected = std::accumulate(padded_image_folded_by_depth_.data(),
				       padded_image_folded_by_depth_.data() + padded_image_folded_by_depth_.num_elements(),
				       0.f);

  BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001f);
  float l2norm = anyfold::l2norm(padded_output_.data(), padded_image_folded_by_depth_.data(),  padded_output_.num_elements());
  BOOST_REQUIRE_CLOSE(l2norm, 0, .00001);
}

BOOST_AUTO_TEST_CASE( all1_convolve )
{


  anyfold::cpu::convolve_3d(padded_image_.data(),(int*)&padded_image_shape_[0],
			    all1_kernel_.data(),&kernel_dims_[0],
			    padded_output_.data());

  float sum = std::accumulate(padded_output_.data(),
			      padded_output_.data() + padded_output_.num_elements(),
			      0.f);
  float sum_expected = std::accumulate(padded_image_folded_by_all1_.data(),
				       padded_image_folded_by_all1_.data() + padded_image_folded_by_all1_.num_elements(),
				       0.f);

  BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001f);
  float l2norm = anyfold::l2norm(padded_output_.data(), padded_image_folded_by_all1_.data(),  padded_output_.num_elements());
  BOOST_REQUIRE_CLOSE(l2norm, 0, .00001);
}
BOOST_AUTO_TEST_SUITE_END()
