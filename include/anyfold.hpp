#ifndef _ANYFOLD_H_
#define _ANYFOLD_H_

#include "cpu/convolve.hpp"

#ifdef HAS_OPENCL
#include "opencl/convolve.hpp"
#include "gpu/convolve.hpp"
#endif /* HAS_OPENCL */

#endif /* _ANYFOLD_H_ */
