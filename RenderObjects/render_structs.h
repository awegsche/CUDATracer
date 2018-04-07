#pragma once
#include <vector_types.h>
#include <cuda_runtime.h>
#include <math.h>
#include "Materials/Materialmanager.h"
//#include <cuda_>

struct Ray {
	float3 o, d;
};

#define BLOCKDIM_X 32
#define BLOCKDIM_Y 32

__host__ __device__  inline float3 _make_float3(const float x, const float y, const float z) {
	float3 f;
	f.x = x;
	f.y = y;
	f.z = z;
	return f;
}

// multiplication with scalar
__host__ __device__ inline float3 operator*(const float a, const float3 &v) {
	float3 u;
	u.x = v.x * a;
	u.y = v.y * a;
	u.z = v.z * a;
	return u;
}

// multiplication with scalar
__host__ __device__ inline float3 operator*(const float3 &v, const float a) {
	float3 u;
	u.x = v.x * a;
	u.y = v.y * a;
	u.z = v.z * a;
	return u;
}

// addition
__host__ __device__ inline float3 operator+(const float3 &u, const float3 &v) {
	float3 w;
	w.x = v.x + u.x;
	w.y = v.y + u.y;
	w.z = v.z + u.z;
	return w;
}

// dot product
__host__ __device__ inline float operator*(const float3 &v, const float3 &u) {
	return v.x * u.x + v.y * u.y + v.z * u.z;
}

// cross product
__host__ __device__ inline float3 operator^(const float3 &u, const float3 &v) {
	float3 w;
	w.x = u.y * v.z - u.z * v.y;
	w.y = u.z * v.x - u.x * v.z;
	w.z = u.x * v.y - u.y * v.x;

	return w;
}

__host__ __device__ inline float3 operator-(const float3 &v, const float3 &u) {
	float3 w;
	w.x = v.x - u.x;
	w.y = v.y - u.y;
	w.z = v.z - u.z;
	return w;
}

__host__ __device__ inline float3 operator-(const float3 &rhs) {
	float3 ret;
	ret.x = rhs.x;
	ret.y = rhs.y;
	ret.z = rhs.z;
	return ret;
}

__host__ __device__ inline float4 &operator+=(float4 &u, const float4 &v) {

	u.x += v.x;
	u.y += v.y;
	u.z += v.z;
	u.w += v.w;
	return u;
}

__host__ __device__ inline float3 &operator+=(float3 &u, const float3 &v) {

	u.x += v.x;
	u.y += v.y;
	u.z += v.z;
		
	return u;
}



__host__ __device__ inline float _get_length(const float3 &v) {
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ inline float3 _normalize(const float3 &v) {
	float one_over_l = 1.0f / _get_length(v);
	return one_over_l * v;
}

__device__ inline float clamp(const float f, const float min, const float max)
{
	return f < min ? min : f > max ? max : f;
}

__device__ inline bool _inside_bb(const float3 &p, const float3 &bb_p0, const float3 &bb_p1) {
	return p.x > bb_p0.x && p.x < bb_p1.x
		&& p.y > bb_p0.y && p.y < bb_p1.y
		&& p.z > bb_p0.z && p.z < bb_p1.z;
}


struct ShadeRec {
	float3 normal;
	int hdir;
	float u;
	float v;
	float t;
	Ray ray;
	material_pos material;

	__device__ float3 hitPoint() {
		return ray.o + ray.d * t;
	}

};
