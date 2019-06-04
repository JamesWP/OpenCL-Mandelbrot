
float2 multiply(float2 a, float2 b) {
    float2 mul = { a.s0*b.s0-a.s1*b.s1, a.s1*b.s0+a.s0*b.s1 };
	return mul;
}

float norm(int i, float2 z, int max_iterations) {
    return i + (log(log(convert_float(max_iterations))) - log(log(length(z))))/log(2.0);
}

float norm_mandelbrot(const float2 c)
{
	const size_t max_iterations = 100;

	float2 z = { 0.0, 0.0 };	
	
	size_t i;

	for(i = 0; i < max_iterations; i++)
	{
		z = multiply(z, z) + c;
		if(length(z) > 4.0) 
			return norm(i, z, max_iterations);
	}

	return -1.0;
}

// Mandelbrot kernel
__kernel void mandelbrot(__global float* reals,
	                     __global float* imags,
	                     __write_only image2d_t image)
{
	const float3 white = { 0.0, 0.0, 0.0 };

	const size_t num_colors = 6;
	const float3 cols[num_colors] = {
		{ 0.0, 0.0, 1.0 },
		{ 0.0, 1.0, 1.0 },
		{ 0.0, 1.0, 0.0 },
		{ 1.0, 1.0, 0.0 },
		{ 1.0, 0.0, 0.0 },
		{ 1.0, 0.0, 1.0 }
	};

	const float2 c = { reals[get_global_id(0)], imags[get_global_id(1)] };

	const float norm_mb = norm_mandelbrot(c);

	// is point in mandelbrot set?
	const bool mb = norm_mb < 0.0;

	const int whole = convert_int(floor(norm_mb)) % num_colors;
	const int next_whole = (whole + 1) % num_colors;
	
	const float frac = norm_mb - floor(norm_mb);
		
	// calculate color
	const float3 color = mb ? white : mix(cols[whole], cols[next_whole], frac);

	const uint3 colorUI = convert_uint3(color * 255.0f);
	const uint4 colorWithAlpha = { colorUI, 0u };

	const int2 coord = { get_global_id(0), get_global_id(1) };
	
	write_imageui(image, coord, colorWithAlpha);
}