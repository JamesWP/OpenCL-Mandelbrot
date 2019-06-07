#pragma once
#include "mandelbrot.h"

namespace raster {
	void write_output(std::string filename, const mandelbrot::host_output& output);
	void write_output(std::ostream& out, const mandelbrot::host_output& output);
}