#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__constant sampler_t sampler =
	CLK_NORMALIZED_COORDS_FALSE
	| CLK_ADDRESS_CLAMP
	| CLK_FILTER_NEAREST;

float currentWeight (
	__constant const float* filterWeights,
	const int x, const int y, const int z)
{
	int xx = (FILTER_SIZE_X-1 - (x+1));
	int yy = (FILTER_SIZE_Y-1 - (y+1));
	int zz = (FILTER_SIZE_Z-1 - (z+1));
	if(xx < 0 || yy < 0 || zz < 0)
	{
		return 0.0f;
	}

	return filterWeights[xx +
	                     yy * FILTER_SIZE_X +
	                     zz * FILTER_SIZE_Y * FILTER_SIZE_X];
}

__kernel void convolution3d (__global float* input,
                             __constant float* filterWeights,
                             __global float* inter,
                             __global float* output,
                             int3 offset)
{
	__local float values[6*6*6];

	int4 pos = {get_global_id(0),
	            get_global_id(1),
	            get_global_id(2),
	            0};
	int gidx = pos.z * (get_global_size(1)) * (get_global_size(0)) +
		   pos.y * (get_global_size(0)) +
		   pos.x;
	int4 pos2 = {-FILTER_SIZE_X_HALF + offset.x + 1 + (get_global_id(0)+FILTER_SIZE_X_HALF),
	             -FILTER_SIZE_Y_HALF + offset.y + 1 + (get_global_id(1)+FILTER_SIZE_Y_HALF),
	             -FILTER_SIZE_Z_HALF + offset.z + 1 + (get_global_id(2)+FILTER_SIZE_Z_HALF),
	             0};

	int gidx2 = pos2.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) +
		    pos2.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) +
		    pos2.x;
	int lidx = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
		   (get_local_id(1)+1) * (get_local_size(0)+2) +
		   (get_local_id(0)+1);

	float oldVal = inter[gidx];

	values[lidx] = input[gidx2];

	if(get_local_id(0) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF - 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(1) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF - 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(2) == 0)
	{
		int id = (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF - 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}

	if(get_local_id(0) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (3+2);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF + 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(1) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (3+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF + 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(2) == 3)
	{
		int id = (3+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF + 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}

	if(get_local_id(0) == 0 && get_local_id(1) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF - 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF - 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(0) == 0 && get_local_id(2) == 0)
	{
		int id = (get_local_id(1)+1) * (get_local_size(0)+2);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF - 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF - 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(1) == 0 && get_local_id(2) == 0)
	{
		int id = (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF - 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF - 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}

	if(get_local_id(0) == 3 && get_local_id(1) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (3+2) * (get_local_size(0)+2) +
			 (3+2);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF + 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF + 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(0) == 3 && get_local_id(2) == 3)
	{
		int id = (3+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (3+2);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF + 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF + 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(1) == 3 && get_local_id(2) == 3)
	{
		int id = (3+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (3+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF + 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF + 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}

	if(get_local_id(0) == 0 && get_local_id(1) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (3+2) * (get_local_size(0)+2);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF - 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF + 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(0) == 0 && get_local_id(2) == 3)
	{
		int id = (3+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF - 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF + 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(1) == 0 && get_local_id(2) == 3)
	{
		int id = (3+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF - 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF + 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(0) == 3 && get_local_id(1) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (3+2);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF + 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF - 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(0) == 3 && get_local_id(2) == 0)
	{
		int id = (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (3+2);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF + 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF - 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}
	if(get_local_id(1) == 3 && get_local_id(2) == 0)
	{
		int id = (3+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF + 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF - 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
	}

	if(get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0)
	{
		int id = 0;
		int4 p = {-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF - 1,
		          -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF - 1,
		          -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF - 1,
		          0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];

		id = 5;
		p = (int4){-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF + 4,
		           -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF - 1,
		           -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF - 1,
		           0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];

		id = 5 * (get_local_size(0)+2);
		p = (int4){-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF - 1,
		           -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF + 4,
		           -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF - 1,
		           0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];

		id = 5 * (get_local_size(0)+2) + 5;
		p = (int4){-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF + 4,
		           -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF + 4,
		           -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF - 1,
		           0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];

		id = 5 * (get_local_size(1)+2) * (get_local_size(0)+2);
		p = (int4){-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF - 1,
		           -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF - 1,
		           -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF + 4,
		           0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];

		id = 5 * (get_local_size(1)+2) * (get_local_size(0)+2) + 5;
		p = (int4){-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF + 4,
		           -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF - 1,
		           -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF + 4,
		           0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];

		id = 5 * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     5 * (get_local_size(0)+2);
		p = (int4){-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF - 1,
		           -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF + 4,
		           -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF + 4,
		           0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];

		id = 5 * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     5 * (get_local_size(0)+2) + 5;
		p = (int4){-FILTER_SIZE_X_HALF + offset.x + 1 + get_global_id(0)+FILTER_SIZE_X_HALF + 4,
		           -FILTER_SIZE_Y_HALF + offset.y + 1 + get_global_id(1)+FILTER_SIZE_Y_HALF + 4,
		           -FILTER_SIZE_Z_HALF + offset.z + 1 + get_global_id(2)+FILTER_SIZE_Z_HALF + 4,
		           0};
		values[id] = input[p.z * (get_global_size(1)+2*FILTER_SIZE_Y_HALF) * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.y * (get_global_size(0)+2*FILTER_SIZE_X_HALF) + p.x];
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

	output[gidx] = sum;
}
