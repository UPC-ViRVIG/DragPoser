#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

namespace py = pybind11;

struct quaternion
{
	float w;
	float x;
	float y;
	float z;

	quaternion() : w(1), x(0), y(0), z(0) {}
	quaternion(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}
};

struct float3
{
	float x;
	float y;
	float z;

	float3() : x(0), y(0), z(0) {}
	float3(float x, float y, float z) : x(x), y(y), z(z) {}
};

struct float2
{
	float x;
	float y;

	float2() : x(0), y(0) {}
	float2(float x, float y) : x(x), y(y) {}
};

void add_venv_libs();

py::array_t<float> create_numpy_array_quaternion(quaternion* buffer, int length);
py::array_t<float> create_numpy_array_float3(float3* buffer, int length);
py::array_t<float> create_numpy_array_float2(float2* buffer, int length);
py::array_t<float> create_numpy_array_float(float* buffer, int length);

void logMessage(const std::string& message);