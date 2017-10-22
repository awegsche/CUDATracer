#ifndef CAMERA_CU
#define CAMERA_CU

#include "Camera.h"

#include "Materials/RGBColors.h"
#include "render_structs.h"

#include "MinecraftWorld/MCWorld.h"
#include "MinecraftWorld/MCWorld.cu"
#include "Materials/material.cu"

__device__ float4 trace_ray(
	Ray &ray,
	const float3 &p0, const float3 &p1, const int nx, const int ny, const int nz,
	world_struct *world) 
{
	float t = kHUGEVALUE;
	chunk_struct** cells = world->chunks;
	ShadeRec sr;

	if (world_hit(
		ray, t,
		p0, p1, nx, ny, nz,
		cells, sr, world->blocks)
		) 
	{
		//return rgbcolor(sr.hitPoint.x , sr.hitPoint.y, sr.hitPoint.z );
		return shade(sr, world->materials, world->texels, world->positions, world->dimensions);// -rgbcolor(t / 1000.f, 0.f, 0.f);
		return rgbcolor(1, 0, 0);
	}
	return rgbcolor(ray.d.x, ray.d.y, ray.d.z);
}

// The kernel to render with the Thinlens camera
__global__ void render_kernel(
	uchar4 *dst, const int hres, const int vres, const int num_samples, const float s,
	float3 eye, float3 u, float3 v, float3 w, float aperture, float d, world_struct* world)
{
	Ray ray;

	float2 sp;      // Sample point in [0,1]x[0,1]
	float2 pp;      // Sample point on a pixel
	float2 ap;     // Sample point on aperture;

	sp.x = 0.0f;
	sp.y = 0.0f;
	ap.x = 0.0f;
	ap.y = 0.0f;

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;




	int depth = 0;
	float4 L = rgbcolor(.5, .8, 1.0);


	for (int j = 0; j < num_samples; j++) {
		//sp = vp.sampler_ptr->sample_unit_square();

		pp.x = s * (ix - 0.5 * hres + sp.x);
		pp.y = s * (iy - 0.5 * vres + sp.y);

		//ap = _sampler_ptr->sample_unit_disk();
		ray.o = eye + (aperture * ap.x) * u + (aperture * ap.y) * v;
		float3 dir = (pp.x - aperture * ap.x) * u + (pp.y - aperture * ap.y) * v - d * w;
		ray.d = _normalize(dir);
		L = trace_ray(
			ray,
			world->bb_p0, world->bb_p1, WORLDSIZE_INCHUNKS, 16, WORLDSIZE_INCHUNKS,
			world);
		//L = rgbcolor(ray.d.x, ray.d.y, ray.d.z);

	}
	/*L /= num_samples;
	L *= exposure_time;
	rgb[column] = L.truncate().to_uint();*/
	dst[ix + iy * hres] = _rgbcolor_to_byte(L);

	

}


Camera::Camera()
	:d(1000.0f), zoom(1.0f)
{
	up = make_float3(0.f, 1.f, 0.f);
}


Camera::~Camera()
{
}

void Camera::render(uchar4 * frame, const int width, const int height) const
{
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 num_blocks = dim3(width / BLOCKDIM_X, height / BLOCKDIM_Y);

	float3 f = _make_float3(100, 1, 0);
	float3 norm = _normalize(f);

	float3 eye2 = eye;

	render_kernel << <num_blocks, threads >> > (
		frame, width, height, 1, zoom,
		eye, u, v, w, 1.0f, d, world);


}

void Camera::compute_uvw()
{
	w = eye - lookat;
	w = _normalize(w);
	u = up ^ w;
	u = _normalize(u);
	v = w ^ u;
}

void Camera::set_world(world_struct * w)
{
	world = w;
}

void Camera::set_eye(float x, float y, float z)
{
	eye = _make_float3(x, y, z);
}

void Camera::move_eye(float x, float y, float z)
{
	eye += _make_float3(x, y, z);
}

void Camera::move_eye_forward(float d)
{
	eye += w * (-d);
}

void Camera::set_lookat(float x, float y, float z)
{
	lookat = _make_float3(x, y, z);

}

void Camera::move_eye_left(float d)
{
	eye += u * d;
}

void Camera::move_eye_right(float d)
{
	eye += u * (-d);
}

void Camera::move_eye_backward(float d)
{
	eye += w * (d);
}

void Camera::rotate_up(float d)
{
	w += up * d;
	w = _normalize(w);
	u = up ^ w;
	u = _normalize(u);
	v = w ^ u;
}

void Camera::rotate_down(float d)
{
	w += up * (-d);
	w = _normalize(w);
	u = up ^ w;
	u = _normalize(u);
	v = w ^ u;
}

void Camera::rotate_left(float d)
{
	w += u * d;
	w = _normalize(w);
	u = up ^ w;
	u = _normalize(u);
	v = w ^ u;
}

void Camera::rotate_right(float d)
{
	w += u * (-d);
	w = _normalize(w);
	u = up ^ w;
	u = _normalize(u);
	v = w ^ u;
}


#endif