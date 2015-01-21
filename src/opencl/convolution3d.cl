#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__constant sampler_t sampler =
	CLK_NORMALIZED_COORDS_FALSE
	| CLK_ADDRESS_CLAMP
	/* | CLK_ADDRESS_CLAMP_TO_EDGE */
	| CLK_FILTER_NEAREST;

float currentWeight (__constant const float* filterWeights,
                     const int x, const int y, const int z)
{
	/* return filterWeights[(3-1-(x+1)) + */
	/*                      (3-1-(y+1)) * 3 + */
	/*                      (3-1-(z+1)) * 3 * 3]; */
	return filterWeights[(FILTER_SIZE-1-(x+FILTER_SIZE_HALF)) +
	                     (FILTER_SIZE-1-(y+FILTER_SIZE_HALF)) * FILTER_SIZE +
	                     (FILTER_SIZE-1-(z+FILTER_SIZE_HALF)) * FILTER_SIZE * FILTER_SIZE];
}

__kernel void convolution3d (__read_only image3d_t input,
                             __constant float* filterWeights,
                             __read_only image3d_t inter,
                             __write_only image3d_t output,
                             int3 offset)
{
	__local float values[6*6*6];
	
	int gidx = (-FILTER_SIZE_HALF+1+get_global_id(2)) * get_global_size(1) * get_global_size(0) +
	           (-FILTER_SIZE_HALF+1+get_global_id(1)) * get_global_size(0) +
	           (-FILTER_SIZE_HALF+1+get_global_id(0));


	int4 pos = {(get_global_id(0)),
	            (get_global_id(1)),
	            (get_global_id(2)), 0};
	int4 pos2 = {(-FILTER_SIZE_HALF+1+get_global_id(0)),
	            (-FILTER_SIZE_HALF+1+get_global_id(1)),
	            (-FILTER_SIZE_HALF+1+get_global_id(2)), 0};
	float oldVal = read_imagef(inter, sampler, pos).x;
	/* if(oldVal != 0.0f)printf("%d ", oldVal); */
	int lidx = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
		   (get_local_id(1)+1) * (get_local_size(0)+2) +
		   (get_local_id(0)+1);
	values[lidx] = read_imagef(input, sampler, pos2).x;
	/* printf("%d %f\n", lidx, values[lidx]); */

	if(get_local_id(0) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+0);
		int4 p = {(-FILTER_SIZE_HALF+1+get_global_id(0))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1)),
		          (-FILTER_SIZE_HALF+1+get_global_id(2)), 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub0"); */
		/* printf("%d ", id); */
	}
	if(get_local_id(1) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+0) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0)),
		          (-FILTER_SIZE_HALF+1+get_global_id(1))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2)), 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub1"); */
		/* printf("%d ", id); */
	}
	if(get_local_id(2) == 0)
	{
		int id = (get_local_id(2)+0) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0)),
		                (-FILTER_SIZE_HALF+1+get_global_id(1)),
		                (-FILTER_SIZE_HALF+1+get_global_id(2))-1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub2"); */
		/* printf("%d %f %d %d %d\n", id, values[id], p.x, p.y, p.z); */
	}
	 if(get_local_id(0) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+2);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))+1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1)),
		          (-FILTER_SIZE_HALF+1+get_global_id(2)), 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub00"); */
		/* printf("%d ", id); */
	}
	 if(get_local_id(1) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0)),
		          (-FILTER_SIZE_HALF+1+get_global_id(1))+1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2)), 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub01"); */
		/* printf("%d ", id); */
	}
	 if(get_local_id(2) == 3)
	{
		int id = (get_local_id(2)+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0)),
		          (-FILTER_SIZE_HALF+1+get_global_id(1)),
		          (-FILTER_SIZE_HALF+1+get_global_id(2))+1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub02"); */
		/* printf("%d ", id); */
	}

	if(get_local_id(0) == 0 && get_local_id(1) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+0) * (get_local_size(0)+2) +
			 (get_local_id(0)+0);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2)), 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub"); */
		/* printf("%d ", id); */
	}
	 if(get_local_id(0) == 0 && get_local_id(2) == 0)
	{
		int id = (get_local_id(2)+0) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+0);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1)),
		          (-FILTER_SIZE_HALF+1+get_global_id(2))-1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub022"); */
		/* printf("%d ", id); */
	}
	 if(get_local_id(1) == 0 && get_local_id(2) == 0)
	{
		int id = (get_local_id(2)+0) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+0) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0)),
		          (-FILTER_SIZE_HALF+1+get_global_id(1))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2))-1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub122"); */
		/* printf("%d ", id); */
	}
	 if(get_local_id(0) == 3 && get_local_id(1) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+2);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))+1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1))+1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2)), 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub011"); */
		/* printf("%d ", id); */
	}
	 if(get_local_id(0) == 3 && get_local_id(2) == 3)
	{
		int id = (get_local_id(2)+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+2);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))+1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1)),
		          (-FILTER_SIZE_HALF+1+get_global_id(2))+1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub02"); */
		/* printf("%d ", id); */
	}
	 if(get_local_id(1) == 3 && get_local_id(2) == 3)
	{
		int id = (get_local_id(2)+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0)),
		          (-FILTER_SIZE_HALF+1+get_global_id(1))+1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2))+1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub13"); */
		/* printf("%d ", id); */
	}

	 if(get_local_id(0) == 0 && get_local_id(1) == 3)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+0);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1))+1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2)), 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub"); */
		/* printf("%d ", id); */
	}
	 if(get_local_id(0) == 0 && get_local_id(2) == 3)
	{
		int id = (get_local_id(2)+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+0);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1)),
		          (-FILTER_SIZE_HALF+1+get_global_id(2))+1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub022"); */
		/* printf("%d ", id); */
	}
	 if(get_local_id(1) == 0 && get_local_id(2) == 3)
	{
		int id = (get_local_id(2)+2) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+0) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0)),
		          (-FILTER_SIZE_HALF+1+get_global_id(1))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2))+1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub122"); */
		/* printf("%d ", id); */
	}
	 if(get_local_id(0) == 3 && get_local_id(1) == 0)
	{
		int id = (get_local_id(2)+1) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+0) * (get_local_size(0)+2) +
			 (get_local_id(0)+2);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))+1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2)), 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub011"); */
		/* printf("a%d ", id); */
	}
	 if(get_local_id(0) == 3 && get_local_id(2) == 0)
	{
		int id = (get_local_id(2)+0) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+1) * (get_local_size(0)+2) +
			 (get_local_id(0)+2);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))+1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1)),
		          (-FILTER_SIZE_HALF+1+get_global_id(2))-1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub02"); */
		/* printf("z%d ", id); */
	}
	 if(get_local_id(1) == 3 && get_local_id(2) == 0)
	{
		int id = (get_local_id(2)+0) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(0)+1);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0)),
		          (-FILTER_SIZE_HALF+1+get_global_id(1))+1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2))-1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blub13"); */
		/* printf("x%d ", id); */
	}
	 
	if(get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0)
	{
		int id = (get_local_id(2)+0) * (get_local_size(1)+2) * (get_local_size(0)+2) +
			 (get_local_id(1)+0) * (get_local_size(0)+2) +
			 (get_local_id(0)+0);
		int4 p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(1))-1,
		          (-FILTER_SIZE_HALF+1+get_global_id(2))-1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blubq"); */
		/* printf("%d ", id); */

		id = (get_local_id(2)+0) * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     (get_local_id(1)+0) * (get_local_size(0)+2) +
		     (get_local_id(0)+5);
		p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))+4,
		     (-FILTER_SIZE_HALF+1+get_global_id(1))-1,
		     (-FILTER_SIZE_HALF+1+get_global_id(2))-1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blubw"); */
		/* printf("%d ", id); */

		id = (get_local_id(2)+0) * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     (get_local_id(1)+5) * (get_local_size(0)+2) +
		     (get_local_id(0)+0);
		p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))-1,
		     (-FILTER_SIZE_HALF+1+get_global_id(1))+4,
		     (-FILTER_SIZE_HALF+1+get_global_id(2))-1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blube"); */
		/* printf("%d ", id); */

		id = (get_local_id(2)+0) * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     (get_local_id(1)+5) * (get_local_size(0)+2) +
		     (get_local_id(0)+5);
		p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))+4,
		     (-FILTER_SIZE_HALF+1+get_global_id(1))+4,
		     (-FILTER_SIZE_HALF+1+get_global_id(2))-1, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blubr"); */
		/* printf("%d ", id); */

		id = (get_local_id(2)+5) * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     (get_local_id(1)+0) * (get_local_size(0)+2) +
		     (get_local_id(0)+0);
		p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))-1,
		     (-FILTER_SIZE_HALF+1+get_global_id(1))-1,
		     (-FILTER_SIZE_HALF+1+get_global_id(2))+4, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blubt"); */
		/* printf("%d ", id); */

		id = (get_local_id(2)+5) * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     (get_local_id(1)+0) * (get_local_size(0)+2) +
		     (get_local_id(0)+5);
		p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))+4,
		     (-FILTER_SIZE_HALF+1+get_global_id(1))-1,
		     (-FILTER_SIZE_HALF+1+get_global_id(2))+4, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("bluby"); */
		/* /\* printf("%d ", id); *\/ */

		id = (get_local_id(2)+5) * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     (get_local_id(1)+5) * (get_local_size(0)+2) +
		     (get_local_id(0)+0);
		p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))-1,
		     (-FILTER_SIZE_HALF+1+get_global_id(1))+4,
		     (-FILTER_SIZE_HALF+1+get_global_id(2))+4, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blubu"); */
		/* printf("%d %d %d %d %f\n", id, p.x, p.y, p.z, values[id]); */

		id = (get_local_id(2)+5) * (get_local_size(1)+2) * (get_local_size(0)+2) +
		     (get_local_id(1)+5) * (get_local_size(0)+2) +
		     (get_local_id(0)+5);
		p = (int4){(-FILTER_SIZE_HALF+1+get_global_id(0))+4,
		     (-FILTER_SIZE_HALF+1+get_global_id(1))+4,
		     (-FILTER_SIZE_HALF+1+get_global_id(2))+4, 0};
		values[id] = read_imagef(input, sampler, p).x;
		/* if(isnan(values[id])) printf("blubi"); */
		/* printf("abc %d %d %d %d ", id, get_local_size(0), get_local_size(1), get_local_size(2)); */
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	/* if(get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) */
	/* { */
	/* 	for(int z = 0; z < 6; ++z) */
	/* 	{ */
	/* 		for(int y = 0; y < 6; ++y) */
	/* 		{ */
	/* 			for(int x = 0; x < 6; ++x) */
	/* 			{ */
	/* 				/\* printf("%f %d %d %d ", values[z*6*6+y*6+x], x, y, z); *\/ */
	/* 				printf("%f ", values[z*6*6+y*6+x]); */
	/* 			} */
	/* 			printf("\ny%d.\n", y); */
	/* 		} */
	/* 		printf("\n\nz%d..\n\n",z); */

	/* 		/\* for(int y = 0; y < 6; ++y) *\/ */
	/* 		/\* { *\/ */
	/* 		/\* 	for(int x = 0; x < 6; ++x) *\/ */
	/* 		/\* 	{ *\/ */
	/* 		/\* 		printf("%f ", read_imagef(input, sampler, ); *\/ */
	/* 		/\* 	}printf("\n"); *\/ */
	/* 		/\* }printf("\n\n"); *\/ */
	/* 	} */
		
	/* } */
	/* printf("%d %d %d\n", offset.x, offset.y, offset.z); */
	float sum = oldVal;
	for(int z = -1; z <= 1; z++)
	{
		int idz = (get_local_id(2)+1+z) * (get_local_size(1)+2) * (get_local_size(0)+2);
		for(int y = -1; y <= 1; y++)
		{
			int idy = (get_local_id(1)+1+y) * (get_local_size(0)+2);
			for(int x = -1; x <= 1; x++)
			{
				int id = idz + idy + get_local_id(0)+1+x;
				float val = currentWeight(filterWeights, offset.x+x, offset.y+y, offset.z+z)
				/* float val = currentWeight(filterWeights, x, y, z) */
					* values[id];
				sum += val;
				/* sum += currentWeight(filterWeights, x, y, z) */
                                       /* * read_imagef(input, sampler, pos + (int4)(x,y,z,0)).x; */

				/* printf("%d %f %f\n", id, values[id], read_imagef(input, sampler, pos+(int4)(x,y,z,0)).x); */
			}
		}
	}
	/* for(int i = 0; i < 6*6*6; ++i) */
	/* { */
	/* 	if(isnan(values[i])) 	printf("b %d %d %d %d %d %d %d %f\n", i, (-FILTER_SIZE_HALF+1+get_global_id(0)), (-FILTER_SIZE_HALF+1+get_global_id(1)), (-FILTER_SIZE_HALF+1+get_global_id(2)), get_local_id(0), get_local_id(1), get_local_id(2), values[i]); */

	/* } */

	write_imagef(output, pos, sum);
}
