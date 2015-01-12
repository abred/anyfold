//sampler prevents seg faults while reading an image at border
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

void kernel convolve_simple(__read_only image3d_t _image, global const int* _kernel_extends,global const float* _kernel,
			      global float* _result) { 
    //image pixel position
    const int4 image_pos = {get_global_id(0), get_global_id(1), get_global_id(2),0};
    
    _result[image_pos.x + (get_global_size(1) * (image_pos.y + get_global_size(2) * image_pos.z))] = 0;
    
    float image_value = 0;    
    float kernel_value = 0;    
    float value = 0; 
    
    for(int x = -_kernel_extends[0]; x < _kernel_extends[0] + 1; x++) {
        for(int y = -_kernel_extends[1]; y < _kernel_extends[1] + 1; y++) {
	    for(int z = -_kernel_extends[2];z < _kernel_extends[2] + 1; z++) {
	      kernel_value  = _kernel[x+(_kernel_extends[1]*(y+_kernel_extends[2]*z))];
	      image_value = read_imagef(_image, sampler, image_pos + (int4)(x,y,z,0)).x;
	      value += kernel_value * image_value;
	    }
        }
    }
    //Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
   _result[image_pos.x + (get_global_size(1) * (image_pos.y + get_global_size(2) * image_pos.z))] = value; 
} 