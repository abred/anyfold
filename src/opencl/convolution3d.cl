#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

__constant sampler_t sampler =
	CLK_NORMALIZED_COORDS_FALSE
  	| CLK_ADDRESS_CLAMP
	| CLK_FILTER_NEAREST;

float currentWeight (__constant const float* filterWeights,
                     const int x, const int y, const int z)
{
	/* return filterWeights[(x+FILTER_SIZE_HALF) + */
	/*                      (y+FILTER_SIZE_HALF) * FILTER_SIZE + */
	/*                      (z+FILTER_SIZE_HALF) * FILTER_SIZE * FILTER_SIZE]; */
	return filterWeights[(FILTER_SIZE_X-1-(x+FILTER_SIZE_HALF_X)) +
	                     (FILTER_SIZE_Y-1-(y+FILTER_SIZE_HALF_Y)) * FILTER_SIZE_X +
	                     (FILTER_SIZE_Z-1-(z+FILTER_SIZE_HALF_Z)) * FILTER_SIZE_X * FILTER_SIZE_Y];
}

__kernel void convolution3d (__global float* input,
                             __constant float* filterWeights,
                             __global float* output)
{

	const long4 pos = {get_global_id(0),
	                  get_global_id(1),
	                  get_global_id(2), 0};
			  
	const long gidx = get_global_id(0) + get_global_size(0) * ( get_global_id(1)+ get_global_size(1) * get_global_id(2));
	
	printf("sizes: %d,%d,%d\n" ,get_global_size(0),get_global_size(1),get_global_size(2));
	
	//padding
	if (get_global_id(0) < FILTER_SIZE_HALF_X ||
	    get_global_id(0) > IMAGE_SIZE_X - FILTER_SIZE_HALF_X - 1 ||
	    get_global_id(1) < FILTER_SIZE_HALF_Y ||
	    get_global_id(1) > IMAGE_SIZE_Y - FILTER_SIZE_HALF_Y - 1 ||
	    get_global_id(2) < FILTER_SIZE_HALF_Z ||
	    get_global_id(2) > IMAGE_SIZE_Z - FILTER_SIZE_HALF_Z - 1
		)
	{
		output[gidx] = 0;
		printf("kernel_ids: %d,%d,%d %d padded\n", get_global_id(0), get_global_id(1), get_global_id(2), gidx);
		return;
	}

	float sum = 0.0f;
	for(int z = -FILTER_SIZE_HALF_Z; z <= FILTER_SIZE_HALF_Z; z++) {
		long idz = (pos.z+z) * get_global_size(1) * get_global_size(0);
		for(int y = -FILTER_SIZE_HALF_Y; y <= FILTER_SIZE_HALF_Y; y++) {
			long idy = (pos.y+y) * get_global_size(0);
			for(int x = -FILTER_SIZE_HALF_X; x <= FILTER_SIZE_HALF_X; x++) {
				long id = idz + idy + pos.x+x;
				float val = currentWeight(filterWeights, x, y, z)
				       * input[id];
				sum += val;
			}
		}
	}
	
	printf("kernel_ids: %d,%d,%d %d \n", get_global_id(0), get_global_id(1), get_global_id(2), gidx);
	
	output[gidx] = sum;
}
