#pragma once

#include <vector>
#include <complex>

namespace mandelbrot {
	struct input_spec {
		std::complex<float> center;
		float imag_height;
		float real_width;
		int itterations;
		float imag_step;
		float real_step;
	};
	struct host_input {
		std::vector<float> reals, imags;

		host_input(const input_spec& spec);
	};
	struct host_output {
		size_t width, height;
		std::vector<uint32_t> out;

		host_output(const host_input& input);
		host_output(size_t width, size_t height);

		const uint32_t& at(size_t x, size_t y) const { return out[y*width + x]; }
		uint32_t& at(size_t x, size_t y) { return out[y*width + x]; }
	};
}
namespace util {
	inline std::vector<float> gen_values(float middle, float span, float step)
	{
		std::vector<float> out;

		out.clear();
		const int size = (int)(span / step);
		out.reserve(size);

		const float half = span / 2.0f;
		for (float min = middle - half, max = middle + half; min < max; min += step)
		{
			out.push_back(min);
		}

		return out;
	}
}

inline mandelbrot::host_input::host_input(const mandelbrot::input_spec& spec)
	: imags(util::gen_values(spec.center.imag(), spec.imag_height, spec.imag_step))
	, reals(util::gen_values(spec.center.real(), spec.real_width, spec.real_step))
{}

inline mandelbrot::host_output::host_output(const mandelbrot::host_input& input)
	: width{ input.reals.size() }, height{ input.imags.size() }, out(width*height, 0u)
{}

inline mandelbrot::host_output::host_output(size_t _width, size_t _height)
	: width{ _width }, height{ _height }, out(width*height, 0u)
{}