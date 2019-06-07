# CMakeList.txt : CMake project for Mandelbrot, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.12)

project(Mandelbrot)

# For GPU code
find_package(OpenCL REQUIRED)

# For testing
enable_testing()
find_package(GTest MODULE REQUIRED)

# for PNG writing
find_path(LIBGD_INCLUDE NAMES gd.h)
find_library(LIBGD_LIBRARY NAMES libgd)

# Extra code for libgd c++ support (next version of the library will have this included)
set(LIBGD_EXTRA gdpp_extra/gd_io_stream.cxx)
set(LIBGD_INCLUDE ${LIBGD_INCLUDE} gdpp_extra)

# Object Library for common code
add_library(MandelbrotLib OBJECT raster.cpp gpu_compute.cpp ${LIBGD_EXTRA})
target_include_directories(MandelbrotLib PUBLIC ${LIBGD_INCLUDE} ${OpenCL_INCLUDE_DIR} .)
target_link_libraries(MandelbrotLib PUBLIC ${LIBGD_LIBRARY} ${OpenCL_LIBRARY})

# Main executable
add_executable (Mandelbrot main.m.cpp)
target_link_libraries(Mandelbrot PRIVATE MandelbrotLib)

# OpenCL Kernel source
add_custom_target(MandelbrotKernel)
add_custom_command(
        TARGET MandelbrotKernel POST_BUILD
		DEPENDS ${CMAKE_SOURCE_DIR}/mandelbrot.cl
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/mandelbrot.cl
                ${CMAKE_CURRENT_BINARY_DIR}/mandelbrot.cl)
add_dependencies(Mandelbrot MandelbrotKernel)

# Unit test executable
add_executable(MandelbrotUnit raster.g.cpp gpu_compute.g.cpp)
target_link_libraries(MandelbrotUnit PUBLIC MandelbrotLib GTest::GTest GTest::Main)
add_test(MandelbrotUnitTests MandelbrotUnit)

