
float2 multiply(float2 a, float2 b) {
    float2 mul = { a.s0*b.s0-a.s1*b.s1, a.s1*b.s0+a.s0*b.s1 };
	return mul;
}

float norm(int i, float2 z, int max_iterations) {
    return i + (log(log(convert_float(max_iterations))) - log(log(length(z))))/log(2.0);
}

float3 hsvtorgb(float3 hsv)
{  
	float3 i;
	float4 K = { 1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0 };
	float3 six = { 6.0, 6.0, 6.0 };
    float3 p = fabs(fract(hsv.xxx + K.xyz, &i) * six - K.www);
    return hsv.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), hsv.y);
}

float norm_mandelbrot(float2 c)
{
	const size_t max_iterations = 1000;

#define MANDELBROT

#ifdef JULIA
	float2 z = c;	
	float2 zero = { 0.4, 0.4 };
	c = zero;
#endif

#ifdef MANDELBROT
	float2 z = { 0.0, 0.0 };
#endif

	size_t i;

	for(i = 0; i < max_iterations; i++)
	{
		z = multiply(z, z) + c;
		if(fast_length(z) > 4.0) 
			return norm(i, z, max_iterations);
	}

	return -1.0;
}

float3 valuetohsv(float value)
{
	// is point in mandelbrot set?
	if (value < 0.0) {
		const float3 black = { 0.0, 0.0, 0.0 };
		return black;
	}

	const int inum_colors = 2000;
	const float fnum_colors = convert_int(inum_colors);

	float fwhole;
	float frac = fract(value, &fwhole);

	const int whole = convert_int(fwhole) % inum_colors;
	const int next_whole = (whole + 1) % inum_colors;

	float3 col1 = { whole / fnum_colors, 1.0, 1.0 };
	float3 col2 = { next_whole / fnum_colors, 1.0, 1.0 };
	
	return mix(col1, col2, frac);
}

// Mandelbrot kernel
__kernel void mandelbrot(__global float* reals,
	                     __global float* imags,
	                     __write_only image2d_t image)
{
	const float2 c = { reals[get_global_id(0)], imags[get_global_id(1)] };

	const float norm_mb = norm_mandelbrot(c);

	const float3 colorHSV = valuetohsv(norm_mb);
	
	const float3 colorRGB = hsvtorgb(colorHSV);

	const uint3 colorUI = convert_uint3(colorRGB * 255.0f);
	const uint4 colorWithAlpha = { colorUI, 0u };

	const int2 coord = { get_global_id(0), get_global_id(1) };
	
	write_imageui(image, coord, colorWithAlpha);
}