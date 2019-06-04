#include <iostream>
#include <fstream>

#include "mandelbrot.h"
#include "gpu_compute.h"
#include "raster.h"

void pause();

int main(int argc, char* argv[])
{
	try {
		mandelbrot::input_spec spec;
		spec.center = { -0.5f, 0.0f };
		spec.imag_height = 2.0f;
		spec.real_width = 2.0f;
		spec.imag_step = 0.0005f;
		spec.real_step = 0.0005f;

		compute::compute_io_data data{ spec };
		compute::gpu_context context;

		compute::compute(data, context);

		raster::write_output("out2.png", data.output);
	}
	catch (const std::exception& exception) {
		std::cerr << "Error occured when running " << argv[0] << '\n';

		std::cerr << exception.what() << '\n';

		pause();

		return 1;
	}

	return 0;
}

void pause()
{
	std::string done;
	std::getline(std::cin, done);
}