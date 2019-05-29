
#include "gpu_compute.h"

#include <vector>
#include <complex>
#include <array>

#include <CL/cl.h>

namespace impl {
	// gpu buffer helper
	template<typename T>
	class gpu_buffer {
	public:
		cl_mem buff{ 0 };
		void allocate(cl_context context, size_t n, size_t flag = CL_MEM_READ_ONLY);
		void load(cl_command_queue, T* data, size_t size);

		~gpu_buffer();
	};

	class gpu_queue {
	public:
		cl_command_queue queue{ 0 };

		gpu_queue(cl_context context, cl_device_id deviceId);
		~gpu_queue();
	};

	// gpu context helper
	std::pair<cl_context, cl_device_id> get_cl_context() {
		cl_uint platformIdCount = 0;
		clGetPlatformIDs(0, nullptr, &platformIdCount);

		if (platformIdCount == 0) {
			throw std::exception("no platorms found");
		}

		// load details of platforms
		std::vector<cl_platform_id> platformIds(platformIdCount, 0);
		clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

		// assume first platform
		cl_platform_id platformId = platformIds[0];

		cl_uint deviceIdCount = 0;
		clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr,
			&deviceIdCount);

		if (deviceIdCount == 0) {
			throw std::exception("no devices found");
		}

		std::vector<cl_device_id> deviceIds(deviceIdCount);
		clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, deviceIdCount,
			deviceIds.data(), nullptr);

		cl_device_id deviceId = deviceIds[0];

		const cl_context_properties contextProperties[] =
		{
			CL_CONTEXT_PLATFORM,
			reinterpret_cast<cl_context_properties> (platformIds[0]),
			0, 0
		};

		cl_int error = CL_SUCCESS;

		cl_context context = clCreateContext(
			contextProperties, deviceIdCount,
			deviceIds.data(), nullptr,
			nullptr, &error);

		if (CL_SUCCESS != error) {
			throw std::exception("no devices found");
		}

		return std::make_pair(context, deviceId);
	}
}

namespace compute {
	class gpu_context_impl {
	public:
		cl_context context{ 0 };
		cl_device_id deviceId{ 0 };

		gpu_context_impl();
		~gpu_context_impl();
	};
}

// gpu buffer helper
template<typename T>
void impl::gpu_buffer<T>::allocate(cl_context context, size_t n, size_t flag)
{
	cl_int error = CL_SUCCESS;

	buff = clCreateBuffer(context,
		flag,
		sizeof(T) * (n),
		nullptr, &error);

	if (CL_SUCCESS != error) {
		throw std::exception("device buffer alloc failed");
	}
}

template<typename T>
void impl::gpu_buffer<T>::load(cl_command_queue queue, T * data, size_t size)
{
	cl_int error = clEnqueueWriteBuffer(queue, buff, CL_TRUE /* blocking write */, 0u, sizeof(T) * size, data, 0u, NULL, NULL);

	if (CL_SUCCESS != error) {
		throw std::exception("buffer write failed");
	}
}

template<typename T>
impl::gpu_buffer<T>::~gpu_buffer() {
	if (0 != buff) {
		clReleaseMemObject(buff);
	}
}

impl::gpu_queue::gpu_queue(cl_context context, cl_device_id deviceId)
{
	cl_int error = CL_SUCCESS;

	queue = clCreateCommandQueue(context, deviceId, 0, &error);

	if (CL_SUCCESS != error) {
		throw std::exception("device command queue alloc failed");
	}
}

impl::gpu_queue::~gpu_queue() {
	if (0 != queue) {
		clReleaseCommandQueue(queue);
	}
}

// gpu context implementation, details
compute::gpu_context_impl::gpu_context_impl()
{
	
	auto ctx = impl::get_cl_context();

	context = ctx.first;
	deviceId = ctx.second;
}
compute::gpu_context_impl::~gpu_context_impl() {
	if (0 != context) {
		clReleaseContext(context);
	}
}


// gpu context, client interface
compute::gpu_context::gpu_context() :_impl(std::make_unique<gpu_context_impl>()) {}
compute::gpu_context::~gpu_context() = default;

// mandelbrot computation
namespace {
	void compute_impl(compute::compute_io_data& data, compute::gpu_context_impl&);
}

void compute::compute(compute::compute_io_data& data, compute::gpu_context& context)
{
	compute_impl(data, context.impl());
}

namespace {
	// mandelbrot implementation
	void compute_impl(compute::compute_io_data& data, compute::gpu_context_impl& impl)
	{
		impl::gpu_buffer<float> device_reals, device_imags;
		impl::gpu_queue queue{ impl.context, impl.deviceId };

		device_reals.allocate(impl.context, data.input.reals.size(), CL_MEM_READ_ONLY);
		device_imags.allocate(impl.context, data.input.imags.size(), CL_MEM_READ_ONLY);

		device_reals.load(queue.queue, data.input.reals.data(), data.input.reals.size());
		device_imags.load(queue.queue, data.input.imags.data(), data.input.imags.size());
	}
}