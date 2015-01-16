#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__constant sampler_t sampler =
	CLK_NORMALIZED_COORDS_FALSE
	| CLK_ADDRESS_CLAMP
	/* | CLK_ADDRESS_CLAMP_TO_EDGE */
	| CLK_FILTER_NEAREST;

float currentWeight (__constant const float* filterWeights,
                     const int x, const int y, const int z)
{
	/* return filterWeights[(x+FILTER_SIZE_HALF) + */
	/*                      (y+FILTER_SIZE_HALF) * FILTER_SIZE + */
	/*                      (z+FILTER_SIZE_HALF) * FILTER_SIZE * FILTER_SIZE]; */
	return filterWeights[(FILTER_SIZE-1-(x+FILTER_SIZE_HALF)) +
	                     (FILTER_SIZE-1-(y+FILTER_SIZE_HALF)) * FILTER_SIZE +
	                     (FILTER_SIZE-1-(z+FILTER_SIZE_HALF)) * FILTER_SIZE * FILTER_SIZE];
}

__kernel void convolution3d (__global float* input,
                             __constant float* filterWeights,
                             __global float* output)
{

	const int4 pos = {get_global_id(0),
	                  get_global_id(1),
	                  get_global_id(2), 0};
	int gidx = pos.z * get_global_size(1) * get_global_size(0) +
	           pos.y * get_global_size(0) +
	           pos.x;

	if (get_global_id(0) < FILTER_SIZE_HALF ||
	    get_global_id(0) > IMAGE_SIZE - FILTER_SIZE_HALF - 1 ||
	    get_global_id(1) < FILTER_SIZE_HALF ||
	    get_global_id(1) > IMAGE_SIZE - FILTER_SIZE_HALF - 1 ||
	    get_global_id(2) < FILTER_SIZE_HALF ||
	    get_global_id(2) > IMAGE_SIZE - FILTER_SIZE_HALF - 1
		)
	{
		output[gidx] = 0;
		return;
	}

	float sum = 0.0f;
	for(int z = -FILTER_SIZE_HALF; z <= FILTER_SIZE_HALF; z++) {
		int idz = (pos.z+z) * get_global_size(1) * get_global_size(0);
		for(int y = -FILTER_SIZE_HALF; y <= FILTER_SIZE_HALF; y++) {
			int idy = (pos.y+y) * get_global_size(0);
			for(int x = -FILTER_SIZE_HALF; x <= FILTER_SIZE_HALF; x++) {
				int id = idz + idy + pos.x+x;
				float val = currentWeight(filterWeights, x, y, z)
				       * input[id];
				sum += val;
			}
		}
	}
	output[gidx] = sum;
}
