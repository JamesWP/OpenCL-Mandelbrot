
#include "gpu_compute.h"

#include <vector>
#include <complex>
#include <array>
#include <fstream>

#include <CL/cl.h>

namespace impl {

	namespace mem {
		constexpr const size_t r = CL_MEM_READ_ONLY;
		constexpr const size_t w = CL_MEM_WRITE_ONLY;
	}

	template<typename T>
	void CLDeleter(T) {}

	template<>
	void CLDeleter<cl_mem>(cl_mem obj) { if (obj) clReleaseMemObject(obj); }

	template<>
	void CLDeleter<cl_command_queue>(cl_command_queue obj) { if (obj) clReleaseCommandQueue(obj); }

	template<>
	void CLDeleter<cl_program>(cl_program obj) { if (obj) clReleaseProgram(obj); }

	template<>
	void CLDeleter<cl_context>(cl_context obj) { if (obj) clReleaseContext(obj); }

	template<>
	void CLDeleter<cl_kernel>(cl_kernel obj) { if (obj) clReleaseKernel(obj); }

	template<typename T>
	class CLOwner {
		using type = T;

		type val;

	public:
		CLOwner() :val{ 0 } {};

		template<typename Ot>
		CLOwner(CLOwner<Ot>&&) = delete;

		template<typename Ot>
		CLOwner& operator=(CLOwner<Ot>&&) = delete;

		~CLOwner()
		{
			CLDeleter<T>(val);
		}

		T& obj() { return val; }
		const T& obj() const { return val; }
	};

	// gpu buffer helper
	template<typename T, size_t Spec>
	class gpu_buffer : private CLOwner<cl_mem> {
		size_t _size;
	public:
		gpu_buffer(cl_context context, size_t n);

		void load(cl_command_queue, T* data, size_t size);

		size_t size() const { return _size; }

		cl_mem buff() const { return obj(); }
	};

	class gpu_queue : private CLOwner<cl_command_queue> {
	public:
		gpu_queue(cl_context context, cl_device_id deviceId);

		cl_command_queue queue() { return obj(); }
	};

	template<size_t Spec>
	class gpu_image : private CLOwner<cl_mem> {
	public:
		cl_image_desc desc{};

		gpu_image(cl_context, size_t width, size_t height);

		void read(cl_command_queue queue, std::vector<uint32_t>& result);

		cl_mem buff() { return obj(); }
	};

	class gpu_mandelbrot_program : private CLOwner<cl_program> {
	public:
		gpu_mandelbrot_program(cl_context context, cl_device_id deviceId, std::string filename);

		cl_program program() { return obj(); }
	};

	class gpu_mandelbrot_kernel : private CLOwner<cl_kernel> {
	public:
		gpu_mandelbrot_kernel(cl_program program);

		void run(cl_command_queue queue, 
			const gpu_buffer<float, impl::mem::r>& reals, 
			const gpu_buffer<float, impl::mem::r>& imags, 
			gpu_image<impl::mem::w>& result);
	};
}

namespace compute {
	class gpu_context_impl : private impl::CLOwner<cl_context> {
	public:
		cl_device_id deviceId{ 0 };

		gpu_context_impl();

		cl_context context() { return obj(); }
	};
}

// gpu buffer helper
template<typename T, size_t Spec>
impl::gpu_buffer<T, Spec>::gpu_buffer(cl_context context, size_t n)
	:_size(n)
{
	cl_int error = CL_SUCCESS;

	obj() = clCreateBuffer(context,
		Spec,
		sizeof(T) * (n),
		nullptr, &error);

	if (CL_SUCCESS != error) {
		throw std::exception("device buffer alloc failed");
	}
}

template<typename T, size_t Spec>
void impl::gpu_buffer<T, Spec>::load(cl_command_queue queue, T *data, size_t size)
{
	cl_int error = clEnqueueWriteBuffer(queue, obj(), CL_TRUE /* blocking write */, 0u, sizeof(T) * size, data, 0u, NULL, NULL);

	if (CL_SUCCESS != error) {
		throw std::exception("buffer write failed");
	}
}

// gpu queue helper
impl::gpu_queue::gpu_queue(cl_context context, cl_device_id deviceId)
{
	cl_int error = CL_SUCCESS;

	obj() = clCreateCommandQueue(context, deviceId, 0, &error);

	if (CL_SUCCESS != error) {
		throw std::exception("device command queue alloc failed");
	}
}

// gpu image helper
template<size_t Spec>
impl::gpu_image<Spec>::gpu_image(cl_context context, size_t width, size_t height)
{
	cl_int error = CL_SUCCESS;

	cl_image_format format = {};
	format.image_channel_data_type = CL_UNSIGNED_INT8;
	format.image_channel_order = CL_RGBA;

	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = width;
	desc.image_height = height;

	obj() = clCreateImage(context, Spec, &format, &desc, NULL, &error);

	if (CL_SUCCESS != error) {
		throw std::exception("error allocating image");
	}
}

template<size_t Spec>
void impl::gpu_image<Spec>::read(cl_command_queue queue, std::vector<uint32_t>& result)
{
	result.resize(desc.image_width * desc.image_height);

	const size_t origin[] = { 0u, 0u, 0u };
	const size_t region[] = { desc.image_width, desc.image_height, 1u /* depth */ };
	cl_int error = clEnqueueReadImage(queue, obj(), true /* blocking read */, origin, region, 0u, 0u, (void*)result.data(), 0, NULL, NULL);
	if (CL_SUCCESS != error) {
		throw std::exception("Error reading image buff");
	}
}

// gpu mandelbrot program
impl::gpu_mandelbrot_program::gpu_mandelbrot_program(cl_context context, cl_device_id deviceId, std::string filename)
{
	std::ifstream codeStream{ filename };
	if (!codeStream) {
		throw std::runtime_error("could not load kernel from " + filename);
	}
	std::istreambuf_iterator<char> code_begin{ codeStream };
	std::istreambuf_iterator<char> code_end;
	std::string code{ code_begin, code_end };

	cl_int error = CL_SUCCESS;

	const char* codes[] = { code.c_str() };
	const size_t lengths[] = { code.size() };

	cl_program program = clCreateProgramWithSource(context, 1, codes, lengths, &error);

	obj() = program;

	if (CL_SUCCESS != error) {
		throw std::runtime_error("unable to create program from " + filename);
	}

	const cl_device_id devices[] = { deviceId };
	const char* options = "";

	cl_int buildError = clBuildProgram(program, 1, devices, options, NULL, NULL);

	if (CL_SUCCESS != buildError) {
		//TODO extract build error and print
		size_t errorLogLength;
		cl_int error = clGetProgramBuildInfo(program,
			deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &errorLogLength);

		if (CL_SUCCESS != error) {
			throw std::runtime_error("build failed for " + filename + " and error occured getting length of build logs");
		}

		std::string errorLog(errorLogLength, ' ');

		error = clGetProgramBuildInfo(program,
			deviceId, CL_PROGRAM_BUILD_LOG, errorLogLength, (void*)errorLog.data(), NULL);

		if (CL_SUCCESS != error) {
			throw std::runtime_error("build failed for " + filename + " and error occured getting build logs");
		}

		throw std::runtime_error("build failed for " + filename + "\n" + errorLog);
	}
}

impl::gpu_mandelbrot_kernel::gpu_mandelbrot_kernel(cl_program program)
{
	cl_int error = CL_SUCCESS;
	obj() = clCreateKernel(program, "mandelbrot", &error);
	if (CL_SUCCESS != error) {
		throw std::exception("failed to create kernel mandelbrot");
	}
}

void impl::gpu_mandelbrot_kernel::run(cl_command_queue queue, 
	const impl::gpu_buffer<float, impl::mem::r>& reals,
	const impl::gpu_buffer<float, impl::mem::r>& imags, 
	impl::gpu_image<impl::mem::w>& result)
{
	auto setArg = [this, argIdx = 0u](const auto& arg) mutable {
		constexpr const size_t argSize = sizeof(std::remove_reference_t<decltype(arg)>);
		cl_int error = clSetKernelArg(obj(), argIdx++, argSize, (void*)&arg);
		if (CL_SUCCESS != error) {
			throw std::exception("Error setting argument");
		}
	};

	setArg(reals.buff());
	setArg(imags.buff());
	setArg(result.buff());

	constexpr const size_t work_dim = 2;

	const size_t global_work_offset[work_dim] = { 0u, 0u };
	const size_t global_work_size[work_dim] = { reals.size(), imags.size() };
	const size_t local_work_size[work_dim] = { 1u, 1u };

	cl_int error = clEnqueueNDRangeKernel(queue, obj(), work_dim, global_work_offset,
		global_work_size, local_work_size, 0, NULL, NULL);

	if (CL_SUCCESS != error) {
		throw std::runtime_error("Kernel run error: " + std::to_string(error));
	}

	clFlush(queue);
}

// gpu context implementation, details
compute::gpu_context_impl::gpu_context_impl()
{
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

	this->deviceId = deviceIds[0];
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

	obj() = context;
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
		size_t num_reals = data.input.reals.size();
		size_t num_imags = data.input.imags.size();

		// Create input buffers
		impl::gpu_buffer<float, impl::mem::r> device_reals{ impl.context(), num_reals };
		impl::gpu_buffer<float, impl::mem::r> device_imags{ impl.context(), num_imags };

		// Create output buffers
		impl::gpu_image<impl::mem::w> device_result{ impl.context(), num_reals, num_imags };

		// Create context for computation
		impl::gpu_queue queue{ impl.context(), impl.deviceId };
		impl::gpu_mandelbrot_program program{ impl.context(), impl.deviceId, "mandelbrot.cl" };
		impl::gpu_mandelbrot_kernel kernel{ program.program() };

		// Copy input
		device_reals.load(queue.queue(), data.input.reals.data(), num_reals);
		device_imags.load(queue.queue(), data.input.imags.data(), num_imags);

		// Calculate
		kernel.run(queue.queue(), device_reals, device_imags, device_result);

		// Copy result
		device_result.read(queue.queue(), data.output.out);
	}
}