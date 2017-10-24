#ifndef CAMERA_CU
#define CAMERA_CU

#include "Camera.h"

#include "Materials/RGBColors.h"
#include "render_structs.h"

#include "MinecraftWorld/MCWorld.h"
#include "MinecraftWorld/MCWorld.cu"
#include "Materials/material.cu"
#include "Samplers/sampler.cu"

#define D_FACT 1.1f

__device__ float4 trace_ray(
	Ray &ray, world_struct *world, const int index) 
{
	float t = kHUGEVALUE;
	chunk_struct** cells = world->chunks;
	ShadeRec sr;

	if (world_hit(ray, t, world, sr)) 
	{
		//return rgbcolor(sr.hitPoint.x , sr.hitPoint.y, sr.hitPoint.z );
		return shade(sr, world, index);// -rgbcolor(t / 1000.f, 0.f, 0.f);
		return rgbcolor(1, 0, 0);
	}
	float sky = clamp(ray.d.y * 2.5f, 0.0f, .8f);
	
	//return rgbcolor(.7f , .8f , 1.0f );
	return rgbcolor(.6f - sky, .9f - sky, 1.0f - sky * 0.3);
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
	
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int index = ix + iy * hres;

	ap.x = 0;
	ap.y = 0;
	sp.x = 0;
	sp.y = 0;

	int depth = 0;
	float4 L = rgbcolor(0,0,0);


	for (int j = 0; j < num_samples; j++) {
		
		pp.x = s * (ix - 0.5 * hres);
		pp.y = s * (iy - 0.5 * vres);
		ray.o = eye;
		float3 dir = pp.x * u + pp.y * v - d * w;
		ray.d = _normalize(dir);
		L = add_colors(L, trace_ray(
			ray, world, index));
		//L = rgbcolor(ray.d.x, ray.d.y, ray.d.z);
		

	}
	//L = scale_color(L, 1.0f / num_samples);
	//L *= exposure_time;
	
	dst[index] = _rgbcolor_to_byte(L);

}

__global__ void expose_kernel(
	uchar4 *dst, float4 *colors, const int hres, const int vres, const int num_samples, const float s,
	float3 eye, float3 u, float3 v, float3 w, float aperture, float d, world_struct* world, int seed, const int sample_count)
{
	Ray ray;

	float2 sp;      // Sample point in [0,1]x[0,1]
	float2 pp;      // Sample point on a pixel
	float2 ap;     // Sample point on aperture;
	
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int index = ix + iy * hres;


	//ap = sample_disk(world->smplr, ix);
	ap.x = 0;
	ap.y = 0;

	int depth = 0;
	float4 L = rgbcolor(0, 0, 0);
		
	sp = sample_square(world->smplr, seed + index );
	ap = sample_disk(world->smplr, seed + index);

	pp.x = s * (ix - 0.5 * hres + sp.x);
	pp.y = s * (iy - 0.5 * vres + sp.y);

	ray.o = eye + (aperture * ap.x) * u + (aperture * ap.y) * v;
	float3 dir = (pp.x - aperture * ap.x) * u + (pp.y - aperture * ap.y) * v - d * w;
	ray.d = _normalize(dir);
	L = add_colors(L, trace_ray(
		ray, world, index));
	
	
	
	float resc = 1.0f / sample_count;
	float nminus1 = sample_count - 1;
	float4 aver_nminus1 = scale_color(colors[index], nminus1);
	colors[index] = scale_color( add_colors(aver_nminus1, L), resc);
	dst[index] = _rgbcolor_to_byte(colors[index]);
}



Camera::Camera()
	:d(100.0f), zoom(.05f), aperture(.15f)
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



	render_kernel << <num_blocks, threads >> > (
		frame, width, height, world->num_samples, zoom,
		eye, u, v, w, aperture, d, world);


}
void Camera::expose(uchar4 * frame, float4* colors, const int width, const int height, const int sample_count) const
{
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 num_blocks = dim3(width / BLOCKDIM_X, height / BLOCKDIM_Y);



	expose_kernel << <num_blocks, threads >> > (
		frame, colors, width, height, world->num_samples, zoom,
		eye, u, v, w, aperture, d, world, rand(), sample_count);


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

void Camera::increase_d()
{
	d *= D_FACT;
	zoom *= D_FACT;
}

void Camera::decrease_d()
{
	d /= D_FACT;
	zoom /= D_FACT;
}

void Camera::zoom_in()
{
	zoom *= 1.2f;
}

void Camera::zoom_out()
{
	zoom /= 1.2f;
}

void Camera::increase_aperture()
{
	aperture *= 1.5f;
}


void Camera::decrease_aperture()
{
	aperture /= 1.5f;
}

#endif