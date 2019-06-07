#pragma once

#include <vector>
#include <complex>

namespace mandelbrot {
	struct input_spec {
		std::complex<float> center;
		size_t output_width, output_height;
		float zoom_level{ 0.0 };
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
	inline std::vector<float> gen_values(float middle, size_t steps, float zoom_level)
	{
		std::vector<float> out;
		out.reserve(steps);

		float step = 0.002 * pow(10.0, -zoom_level);

		size_t mid_steps = steps / 2;
		float begin = middle - mid_steps * step;
		for (size_t i = 0; i < steps; i++){
			out.push_back(begin + i * step);
		}

		return out;
	}
}

inline mandelbrot::host_input::host_input(const mandelbrot::input_spec& spec)
	: imags(util::gen_values(spec.center.imag(), spec.output_height, spec.zoom_level))
	, reals(util::gen_values(spec.center.real(), spec.output_width, spec.zoom_level))
{}

inline mandelbrot::host_output::host_output(const mandelbrot::host_input& input)
	: width{ input.reals.size() }, height{ input.imags.size() }, out(width*height, 0u)
{}

inline mandelbrot::host_output::host_output(size_t _width, size_t _height)
	: width{ _width }, height{ _height }, out(width*height, 0u)
{}