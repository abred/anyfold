#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__constant sampler_t sampler =
	CLK_NORMALIZED_COORDS_FALSE
	| CLK_ADDRESS_CLAMP
	| CLK_FILTER_NEAREST;

float currentWeight (__constant const float* filterWeights,
                     const int x, const int y, const int z)
{
	return filterWeights[(FILTER_SIZE-1 - (x+1)) +
	                     (FILTER_SIZE-1 - (y+1)) * FILTER_SIZE +
	                     (FILTER_SIZE-1 - (z+1)) * FILTER_SIZE * FILTER_SIZE];
}

__kernel void convolution3d (__read_only image3d_t input,
                             __constant float* filterWeights,
                             __read_only image3d_t inter,
                             __write_only image3d_t output,
                             int3 offset)
{
	__local float values[6*6*6];

	int4 pos = {get_global_id(0),
	            get_global_id(1),
	            get_global_id(2),
	            0};
	int4 pos2 = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0),
	             -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1),
	             -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2),
	             0};

	float oldVal = read_imagef(inter, sampler, pos).x;
	int lidx = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
		   (get_local_id(1)+1) * (get_local_size(0)+2) +
		   (get_local_id(0)+1);
	values[lidx] = read_imagef(input, sampler, pos2).x;

	if(get_local_id(0) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) - 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1),
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2),
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(1) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0),
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) - 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2),
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(2) == 0)
	{
		int id = (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0),
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1),
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) - 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}

	if(get_local_id(0) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (3+2);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) + 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1),
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2),
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(1) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (3+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0),
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) + 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2),
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(2) == 3)
	{
		int id = (3+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0),
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1),
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) + 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}

	if(get_local_id(0) == 0 && get_local_id(1) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) - 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) - 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2),
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(0) == 0 && get_local_id(2) == 0)
	{
		int id = (get_local_id(1)+1) * (get_local_size(0)+2);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) - 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1),
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) - 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(1) == 0 && get_local_id(2) == 0)
	{
		int id = (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0),
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) - 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) - 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}

	if(get_local_id(0) == 3 && get_local_id(1) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (3+2) * (get_local_size(0)+2) +
			 (3+2);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) + 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) + 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2),
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(0) == 3 && get_local_id(2) == 3)
	{
		int id = (3+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (3+2);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) + 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1),
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) + 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(1) == 3 && get_local_id(2) == 3)
	{
		int id = (3+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (3+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0),
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) + 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) + 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}

	if(get_local_id(0) == 0 && get_local_id(1) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (3+2) * (get_local_size(0)+2);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) - 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) + 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2),
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(0) == 0 && get_local_id(2) == 3)
	{
		int id = (3+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) - 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1),
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) + 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(1) == 0 && get_local_id(2) == 3)
	{
		int id = (3+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0),
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) - 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) + 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(0) == 3 && get_local_id(1) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (3+2);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) + 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) - 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2),
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(0) == 3 && get_local_id(2) == 0)
	{
		int id = (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (3+2);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) + 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1),
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) - 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	if(get_local_id(1) == 3 && get_local_id(2) == 0)
	{
		int id = (3+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0),
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) + 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) - 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;
	}

	if(get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0)
	{
		int id = 0;
		int4 p = {-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) - 1,
		          -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) - 1,
		          -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) - 1,
		          0};
		values[id] = read_imagef(input, sampler, p).x;

		id = 5;
		p = (int4){-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) + 4,
		           -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) - 1,
		           -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) - 1,
		           0};
		values[id] = read_imagef(input, sampler, p).x;

		id = 5 * (get_local_size(0)+2);
		p = (int4){-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) - 1,
		           -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) + 4,
		           -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) - 1,
		           0};
		values[id] = read_imagef(input, sampler, p).x;

		id = 5 * (get_local_size(0)+2) + 5;
		p = (int4){-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) + 4,
		           -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) + 4,
		           -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) - 1,
		           0};
		values[id] = read_imagef(input, sampler, p).x;

		id = 5 * (get_local_size(1)+2) * (get_local_size(0)+2);
		p = (int4){-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) - 1,
		           -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) - 1,
		           -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) + 4,
		           0};
		values[id] = read_imagef(input, sampler, p).x;

		id = 5 * (get_local_size(1)+2) * (get_local_size(0)+2) + 5;
		p = (int4){-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) + 4,
		           -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) - 1,
		           -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) + 4,
		           0};
		values[id] = read_imagef(input, sampler, p).x;

		id = 5 * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     5 * (get_local_size(0)+2);
		p = (int4){-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) - 1,
		           -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) + 4,
		           -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) + 4,
		           0};
		values[id] = read_imagef(input, sampler, p).x;

		id = 5 * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     5 * (get_local_size(0)+2) + 5;
		p = (int4){-FILTER_SIZE_HALF + offset.x + 1 + get_global_id(0) + 4,
		           -FILTER_SIZE_HALF + offset.y + 1 + get_global_id(1) + 4,
		           -FILTER_SIZE_HALF + offset.z + 1 + get_global_id(2) + 4,
		           0};
		values[id] = read_imagef(input, sampler, p).x;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	float sum = oldVal;
	for(int z = -1; z <= 1; z++)
	{
		int idz = (get_local_id(2) + 1 + z) * (get_local_size(1)+2) * (get_local_size(0)+2);
		for(int y = -1; y <= 1; y++)
		{
			int idy = (get_local_id(1) + 1 + y) * (get_local_size(0)+2);
			for(int x = -1; x <= 1; x++)
			{
				int id = idz + idy + get_local_id(0) + 1 + x;
				float val = currentWeight(filterWeights,
				                          offset.x+x,
				                          offset.y+y,
				                          offset.z+z)
				             * values[id];
				sum += val;
			}
		}
	}

	write_imagef(output, pos, sum);
}
