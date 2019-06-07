#include <iostream>
#include <fstream>

#include "mandelbrot.h"
#include "gpu_compute.h"
#include "raster.h"

void pause();

int main(int argc, char* argv[])
{
	try {

		const size_t output_width = 1000;
		const size_t output_height = 1000;

		compute::gpu_context context{ output_width, output_height };

		for (size_t i = 0; i < 100; i++) {
			mandelbrot::input_spec spec;

			spec.center = { -0.743643887037158704752191506114774f, 0.131825904205311970493132056385139f };
			spec.output_width = output_width;
			spec.output_height = output_height;
			spec.zoom_level = (i/100.0) * 5.0 + 1.0;

			compute::compute_io_data data{ spec };

			compute::compute(data, context);

			raster::write_output("out" + std::to_string(i), data.output);
		}

		pause();
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