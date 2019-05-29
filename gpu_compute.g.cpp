
#include "gpu_compute.h"

#include <gtest/gtest.h>

TEST(GPUCompute, Prepare)
{
	mandelbrot::input_spec spec;

	compute::compute_io_data data{ spec };
}

TEST(GPUCompute, PopulatesArgs)
{
	float width = 1.0f;
	float height = 1.0f;

	mandelbrot::input_spec spec;
	spec.center = { 0.0f,0.0f };
	spec.imag_height = height;
	spec.real_width = width;
	spec.imag_step = height / 10.0f;
	spec.real_step = width / 10.0f;

	compute::compute_io_data data{ spec };

	ASSERT_EQ(11u, data.input.imags.size());
	ASSERT_EQ(11u, data.input.reals.size());
}

TEST(GPUCompute, ContextCreate)
{
	compute::gpu_context context;
}