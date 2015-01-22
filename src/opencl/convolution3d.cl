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
	return filterWeights[(FILTER_SIZE_X-1-(x+FILTER_SIZE_X_HALF)) +
	                     (FILTER_SIZE_Y-1-(y+FILTER_SIZE_Y_HALF)) * FILTER_SIZE_X +
	                     (FILTER_SIZE_Z-1-(z+FILTER_SIZE_Z_HALF)) * FILTER_SIZE_X * FILTER_SIZE_Y];
}

__kernel void convolution3d (__read_only image3d_t input,
                             __constant float* filterWeights,
                             __write_only image3d_t output)
{

	const int4 pos = {get_global_id(0),
	                  get_global_id(1),
	                  get_global_id(2), 0};

	float sum = 0.0f;
	for(int z = -FILTER_SIZE_Z_HALF; z <= FILTER_SIZE_Z_HALF; z++)
	{
		for(int y = -FILTER_SIZE_Y_HALF; y <= FILTER_SIZE_Y_HALF; y++)
		{
			for(int x = -FILTER_SIZE_X_HALF; x <= FILTER_SIZE_X_HALF; x++)
			{
				sum += currentWeight(filterWeights, x, y, z)
					* read_imagef(input, sampler, pos + (int4)(x,y,z,0)).x;
			}
		}
	}
	write_imagef (output, pos, sum);
}
