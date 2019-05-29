#include <iostream>
#include <fstream>

#include "mandelbrot.h"
#include "gpu_compute.h"
#include "raster.h"

int main()
{
	mandelbrot::input_spec spec;
	spec.center = { 0.0f, 0.0f };
	spec.imag_height = 10.0f;
	spec.real_width = 10.0f;
	spec.imag_step = 0.1f;
	spec.real_step = 0.1f;

	compute::compute_io_data data{ spec };
	compute::gpu_context context;

	compute::compute(data, context);

	raster::write_output("out2.png", data.output);

	std::string done;
	std::getline(std::cin, done);
}
