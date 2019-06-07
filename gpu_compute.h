#pragma once

#include <mandelbrot.h>

namespace compute {
	class compute_io_data {
	public:
		mandelbrot::host_input input;
		mandelbrot::host_output output;

		compute_io_data(const mandelbrot::input_spec& spec);
	};
	
	// implementation detail
	class gpu_context_impl;

	class gpu_context {
	private:
		std::unique_ptr<gpu_context_impl> _impl;
	public:
		gpu_context(size_t num_reals, size_t num_imags);
		~gpu_context();

		gpu_context_impl& impl() { return *_impl;  }
	};

	void compute(compute_io_data&, gpu_context&);
}

inline compute::compute_io_data::compute_io_data(const mandelbrot::input_spec& spec)
	: input{ spec }, output{ input }
{
}
