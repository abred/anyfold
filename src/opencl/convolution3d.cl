#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__constant sampler_t sampler =
	CLK_NORMALIZED_COORDS_FALSE
	| CLK_ADDRESS_CLAMP
	/* | CLK_ADDRESS_CLAMP_TO_EDGE */
	| CLK_FILTER_NEAREST;

float currentWeight (__constant const float* filterWeights,
                     const int x, const int y, const int z)
{
	return filterWeights[(3-1-(x+1)) +
	                     (3-1-(y+1)) * 3 +
	                     (3-1-(z+1)) * 3 * 3];
}

__kernel void convolution3d (__read_only image3d_t input,
                             __constant float* filterWeights,
                             __read_only image3d_t inter,
                             __write_only image3d_t output,
                             int3 offset)
{
	// work group size: 4x4x4
	// filter size: 3x3x3
	// ->local mem (4+1+1)x6x6
	__local float values[6*6*6];
	
	int gidx = get_global_id(2) * get_global_size(1) * get_global_size(0) +
	           get_global_id(1) * get_global_size(0) +
	           get_global_id(0);

	if (get_global_id(0) < FILTER_SIZE_HALF ||
	    get_global_id(0) > IMAGE_SIZE - FILTER_SIZE_HALF - 1 ||
	    get_global_id(1) < FILTER_SIZE_HALF ||
	    get_global_id(1) > IMAGE_SIZE - FILTER_SIZE_HALF - 1 ||
	    get_global_id(2) < FILTER_SIZE_HALF ||
	    get_global_id(2) > IMAGE_SIZE - FILTER_SIZE_HALF - 1
		)
	{
		printf("outside %d %d %d\n", get_global_id(0), get_global_id(1), get_global_id(2));
		read_imagef(iter, sampler, pos + (int4)(x,y,z,0))
		write_imagef(output, pos, sum);
		return;
	}
	else
	{
		printf("inside %d %d %d\n", get_global_id(0), get_global_id(1), get_global_id(2));
	}


	int lidx = (get_local_id(2)+1) * (get_local_size(1)+1) * (get_local_size(0)+1) +
		   (get_local_id(1)+1) * (get_local_size(0)+1) +
		   (get_local_id(0)+1);

	values[lidx] = input[gidx];
	printf("%d %d\n", lidx, gidx);
	printf("local %d %d %d\n", get_local_id(0), get_local_id(1), get_local_id(2));

	if(get_local_id(0) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+1) * (get_local_size(0)+1) +
			 (get_local_id(1)+1) * (get_local_size(0)+1) +
			 (get_local_id(0)+0);
		int gid = (get_global_id(2)+0) * get_global_size(1) * get_global_size(0) +
		          (get_global_id(1)+0) * get_global_size(0) +
			  (get_global_id(0)-1);
		values[id] = input[gid];
		printf("00 %d %d\n", id, gid);
	}
	else if(get_local_id(1) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+1) * (get_local_size(0)+1) +
			 (get_local_id(1)+0) * (get_local_size(0)+1) +
			 (get_local_id(0)+1);
		int gid = (get_global_id(2)+0) * get_global_size(1) * get_global_size(0) +
		          (get_global_id(1)-1) * get_global_size(0) +
			  (get_global_id(0)+0);
		values[id] = input[gid];
		printf("10 %d %d\n", id, gid);
	}
	else if(get_local_id(2) == 0)
	{
		int id = (get_local_id(2)+0) * (get_local_size(1)+1) * (get_local_size(0)+1) +
			 (get_local_id(1)+1) * (get_local_size(0)+1) +
			 (get_local_id(0)+1);
		int gid = (get_global_id(2)-1) * get_global_size(1) * get_global_size(0) +
		          (get_global_id(1)+0) * get_global_size(0) +
			  (get_global_id(0)+0);
		values[id] = input[gid];				
		printf("20 %d %d\n", id, gid);
	}
	else if(get_local_id(0) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+1) * (get_local_size(0)+1) +
			 (get_local_id(1)+1) * (get_local_size(0)+1) +
			 (get_local_id(0)+2);
		int gid = (get_global_id(2)+0) * get_global_size(1) * get_global_size(0) +
		          (get_global_id(1)+0) * get_global_size(0) +
			  (get_global_id(0)+1);
		values[id] = input[gid];
		printf("03 %d %d\n", id, gid);
	}
	else if(get_local_id(1) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+1) * (get_local_size(0)+1) +
			 (get_local_id(1)+2) * (get_local_size(0)+1) +
			 (get_local_id(0)+1);
		int gid = (get_global_id(2)+0) * get_global_size(1) * get_global_size(0) +
		          (get_global_id(1)+1) * get_global_size(0) +
			  (get_global_id(0)+0);
		values[id] = input[gid];
		printf("13 %d %d\n", id, gid);
	}
	else if(get_local_id(2) == 3)
	{
		int id = (get_local_id(2)+2) * (get_local_size(1)+1) * (get_local_size(0)+1) +
			 (get_local_id(1)+1) * (get_local_size(0)+1) +
			 (get_local_id(0)+1);
		int gid = (get_global_id(2)+1) * get_global_size(1) * get_global_size(0) +
		          (get_global_id(1)+0) * get_global_size(0) +
			  (get_global_id(0)+0);
		values[id] = input[gid];
		printf("23 %d %d\n", id, gid);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0.0f;
	for(int z = -1; z <= 1; z++)
	{
		int idz = (get_local_id(2)+1+z) * (get_local_size(1)+1) * (get_local_size(0)+1);
		for(int y = -1; y <= 1; y++)
		{
			int idy = (get_local_id(1)+1+y) * (get_local_size(0)+1);
			for(int x = -1; x <= 1; x++)
			{
				int id = idz + idy + get_local_id(0)+1+x;
				float val = currentWeight(filterWeights, offset.x+x, offset.y+y, offset.z+z)
				       * values[id];
				sum += val;
			}
		}
	}	

	output[gidx] += sum;
}
