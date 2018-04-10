#ifndef CAMERA_CU
#define CAMERA_CU

#include "Camera.h"

#include "Materials/RGBColors.h"
#include "render_structs.h"

#include "MinecraftWorld/MCWorld.h"
#include "MinecraftWorld/MCWorld.cu"
#include "Materials/material.cu"
#include "Samplers/sampler.cu"

#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#define D_FACT 1.05f
#define MOVE_VALUE 10.0f
#define MAX_DEPTH 5

__device__ float3 trace_ray(
	Ray &ray, world_struct *world, const float3 &sp, int depth = 1) 
{
	float t = kHUGEVALUE;
	ShadeRec sr;

	float sky = clamp(ray.d.y * 2.1f, 0.0f, .8f);
	//float3 sky_color = rgbcolor(.8f - sky, .9f - sky, 1.0f - sky * 0.3);
	//float3 L = rgbcolor(.8f - sky, .9f - sky, 1.0f - sky * 0.3);
	rgbcol L;
	rgbacol texel_color;

	bool hit = world_hit(ray, t, world, sr);

	if (hit)
		L = shade(sr, world, sp, hit, texel_color);
	else
		depth = 10000;
	
	while(!hit && depth < MAX_DEPTH) // if transparent block, continue until non-transparent surface is hit 
	{
		depth++;
		ray.o = sr.hitPoint() + kEPSILON * ray.d;
		t = kHUGEVALUE;
		hit = world_hit(ray, t, world, sr);
		L += shade(sr, world, sp, hit, texel_color);
	}
	
	if(depth == 1)
		__syncthreads();
	shade_shadow(world, sr, sp, L, texel_color, hit);
	if(depth == 1)
		__syncthreads();
	/*
	if (depth < MAX_DEPTH) {
		float3 wi;
		shade_reflection(world, sr, sr.ray.d, wi, L, seed, depth + 1);
	}
	*/
	if (t > world->haze_dist)
	{
		float factor = 1.0f / (1.0f + (t - world->haze_dist) * world->haze_strength);

		for (int i = 0; i < world->haze_attenuation; i++)
			factor *= factor;
		L = add_colors(scale_color(L, factor), scale_color(rgbcolor(.8f - sky, .9f - sky, 1.0f - sky * 0.3), 1.0f - factor));
	}
	
	return L;
		
}

// The kernel to render with the Thinlens camera
__global__ void
//__launch_bounds__(1024, 16)
render_kernel(
	float3 *dst, const int hres, const int vres, const int seed, const float s,
	float3 eye, float3 u, float3 v, float3 w, float aperture, float d, world_struct* world)
{
	for(int block_X = 0; block_X < hres; block_X +=  gridDim.x * blockDim.x)
	{
		int ix = threadIdx.x + blockDim.x * blockIdx.x + block_X;
		int iy = threadIdx.y + blockIdx.y * blockDim.y;

		if (ix < hres) {
			int index = ix + iy * hres;
			Ray ray;
			float2 pp;      // Sample point on a pixel

			float3 L = rgbcolor(0, 0, 0);
			float2 sp = sample_square(world->smplr, seed);
			float3 hemisphere_sp = sample_hemisphere(world->smplr, seed);

			pp.x = s * (ix - 0.5 * hres + sp.x);
			pp.y = s * (iy - 0.5 * vres + sp.y);
			ray.o = eye;
			float3 dir = pp.x * u + pp.y * v - d * w;
			ray.d = _normalize(dir);
			dst[index] = add_colors(dst[index], trace_ray(ray, world, hemisphere_sp));
			__syncthreads();
		}
	}

}

__global__ void finish_kernel(uchar4 *dst, float3 *colors, const int hres, const int vres, const
        float exposure)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int index = ix + iy * hres;

	dst[index] = _rgbcolor_to_byte(scale_color(colors[index], exposure));
}

__global__ void expose_kernel(
	float3 *colors, const int hres, const int vres, const float s,
	float3 eye, float3 u, float3 v, float3 w, float aperture, float d, world_struct* world, int seed)
{
	Ray ray;

	
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int index = ix + iy * hres;

	//float4 L = rgbcolor(0, 0, 0);
	float2 pp;
	float3 ap = sample_hemisphere(world->smplr, seed);

	pp.x = s * (ix - 0.5 * hres + ap.x);
	pp.y = s * (iy - 0.5 * vres + ap.y);

	ray.o = eye + (aperture * ap.x) * u + (aperture * ap.y) * v;
	ray.d = _normalize((pp.x - aperture * ap.x) * u + (pp.y - aperture * ap.y) * v - d * w);

	colors[index] = add_colors(colors[index], trace_ray(ray, world, ap));
}



Camera::Camera()
	:d(100.0f), zoom(.05f), aperture(.15f)
{
	up = make_float3(0.f, 1.f, 0.f);
}


Camera::~Camera()
{
}

void Camera::render(rgbcol* colors, const int width, const int height, const float time) const
{
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	//dim3 num_blocks = dim3(width / BLOCKDIM_X, height / BLOCKDIM_Y);
	dim3 num_blocks = dim3(4,  height / BLOCKDIM_Y);

	world->light_dir = _normalize(make_float3(world->light_dir.x, world->light_dir.y, sin(time * 1.0e-4f)));



	render_kernel << <num_blocks, threads >> > (
		colors, width, height, rand(), zoom,
		eye, u, v, w, aperture, d, world);


}
void Camera::expose(rgbcol* colors, const int width, const int height, const int sample_count) const
{
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 num_blocks = dim3(width / BLOCKDIM_X, height / BLOCKDIM_Y);

	expose_kernel << <num_blocks, threads >> > (
		colors, width, height, zoom,
		eye, u, v, w, aperture, d, world, sample_count);


}

void Camera::finish(uchar4 * frame, rgbcol * colors, const int w, const int h, const int sample_count) const
{
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 num_blocks = dim3(w / BLOCKDIM_X, h / BLOCKDIM_Y);



	finish_kernel << <num_blocks, threads >> > (frame, colors, w, h, 1.0f / sample_count);

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
	w += up * (-d);
	w = _normalize(w);
	u = up ^ w;
	u = _normalize(u);
	v = w ^ u;
}

void Camera::rotate_down(float d)
{
	w += up * (d);
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
