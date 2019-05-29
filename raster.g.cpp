#include "raster.h"

#include <gtest/gtest.h>

TEST(Raster, SimpleImage)
{
	mandelbrot::host_output output{ 100u, 100u };
	output.at(output.width / 2, output.height / 2) = 100;

	std::ostringstream outStream;
	std::ostream& out = outStream;

	raster::write_output(out, output);

	ASSERT_NE(0u, outStream.str().size());
}