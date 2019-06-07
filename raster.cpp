
#include "raster.h"

#include <fstream>
#include <gd_io_stream.h>

void raster::write_output(std::string filename,
	const mandelbrot::host_output& output) {
	std::fstream outStream{ filename + ".tiff", std::ios_base::out | std::ios_base::binary };
	
	if (!outStream) {
		throw std::exception{ "unable to open file" };
	}

	// call helper to write
	write_output(outStream, output);
}

void raster::write_output(std::ostream& out,
	const mandelbrot::host_output& output) {
	
	gdImagePtr im = gdImageCreateTrueColor(output.width, output.height);

	for (size_t y = 0; y < output.height; y++)
	{
		for (size_t x = 0; x < output.width; x++)
		{
			unsigned int color = output.at(x, y);
			gdImageSetPixel(im, x, y, color);
		}
	}
	
	ostreamIOCtx outCtx{ out };
	gdImageTiffCtx(im, &outCtx);
	gdFree(im);
}