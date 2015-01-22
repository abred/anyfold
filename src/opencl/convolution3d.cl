#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__constant sampler_t sampler =
	CLK_NORMALIZED_COORDS_FALSE
	| CLK_ADDRESS_CLAMP
	| CLK_FILTER_NEAREST;

float currentWeight (__constant const float* filterWeights,
                     const int x, const int y, const int z)
{
	return filterWeights[(FILTER_SIZE_X-1-(x+FILTER_SIZE_X_HALF)) +
	                     (FILTER_SIZE_Y-1-(y+FILTER_SIZE_Y_HALF)) * FILTER_SIZE_X +
	                     (FILTER_SIZE_Z-1-(z+FILTER_SIZE_Z_HALF)) * FILTER_SIZE_X * FILTER_SIZE_Y];
}

__kernel void convolution3d (__global float* input,
                             __constant float* filterWeights,
                             __global float* output)
{
	const int4 pos = {get_global_id(0)+FILTER_SIZE_X_HALF,
	                  get_global_id(1)+FILTER_SIZE_Y_HALF,
	                  get_global_id(2)+FILTER_SIZE_Z_HALF, 0};
	int gidx = get_global_id(2) * get_global_size(1) * get_global_size(0) +
	           get_global_id(1) * get_global_size(0) +
	           get_global_id(0);

	float sum = 0.0f;
	for(int z = -FILTER_SIZE_Z_HALF; z <= FILTER_SIZE_Z_HALF; z++)
	{
		int idz = (pos.z+z) * (get_global_size(1) + 2*FILTER_SIZE_Y_HALF) *
		          (get_global_size(0) + 2*FILTER_SIZE_X_HALF);
		for(int y = -FILTER_SIZE_Y_HALF; y <= FILTER_SIZE_Y_HALF; y++)
		{
			int idy = (pos.y+y) * (get_global_size(0) + 2*FILTER_SIZE_X_HALF);
			for(int x = -FILTER_SIZE_X_HALF; x <= FILTER_SIZE_X_HALF; x++)
			{
				int id = idz + idy + pos.x+x;
				float val = currentWeight(filterWeights, x, y, z)
				             * input[id];
				sum += val;
			}
		}
	}
	output[gidx] = sum;
}
