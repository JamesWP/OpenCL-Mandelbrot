
#include "raster.h"

#include <fstream>
#include <gd_io_stream.h>

void raster::write_output(std::string filename,
	const mandelbrot::host_output& output) {
	std::fstream outStream{ filename, std::ios_base::out | std::ios_base::binary };
	
	if (!outStream) {
		throw std::exception{ "unable to open file" };
	}

	// call helper to write
	write_output(outStream, output);
}

void raster::write_output(std::ostream& out,
	const mandelbrot::host_output& output) {
	
	gdImagePtr im = gdImageCreate(output.width, output.height);

	int black = gdImageColorAllocate(im, 0, 0, 0);

	// black is unused, it is the default color
	(void)black;

	int white = gdImageColorAllocate(im, 255, 255, 255);

	for (size_t y = 0; y < output.height; y++)
	{
		for (size_t x = 0; x < output.width; x++)
		{
			unsigned int color = output.at(x, y);
			if (color != 0) {
				gdImageSetPixel(im, x, y, white);
			}
		}
	}
	
	ostreamIOCtx outCtx{ out };
	gdImagePngCtx(im, &outCtx);

	gdFree(im);
}