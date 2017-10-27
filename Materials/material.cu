#include "RenderObjects/render_structs.h"
#include <cuda_runtime.h>
#include "Materialmanager.h"
#include "RGBColors.h"
#include "MinecraftWorld/MCWorld.cu"
#include "Samplers/sampler.h"

__device__ float4 get_color(float4* texels, texture_pos* positions, uint2* dimensions,
	texture_pos pos, float u, float v) 
{
	uint2 dim = dimensions[pos];
	int iv = u * (dim.x-1);
	int iu = v * (dim.y-1);

	texture_pos index = positions[pos] + iu + iv * dim.x;
	return texels[index];
}

__device__ bool shade(ShadeRec &sr, world_struct *world, const int seed, bool hitt, float4 &L, float4 &texel_color) {

	if (!hitt) return;

	float3 wo = -sr.ray.d;

	material_params material = world->materials[sr.material];
	texel_color = get_color(world->texels, world->positions, world->dimensions, material.position, sr.u, sr.v);

	if (material.transparent && texel_color.w < 1.0f)
		return false;

	// ==== Simple Ambient ======
	// lambertian rho
	L = scale_color(texel_color, material.ka * 0.2);
	float3 u, v, w;

	w = sr.normal;
	v = _normalize(w ^ make_float3(-0.0073f, 1.0f, 0.0034f));
	u = v ^ w;
	float3 sp = sample_hemisphere(world->smplr, seed);

	Ray shadowray;
	shadowray.o = sr.hitPoint + kEPSILON * sr.normal;
	shadowray.d = sp.x * u + sp.y * v + sp.z * w;
	ShadeRec dum;
	float tshadow = kHUGEVALUE;
	if (!world_hit(shadowray, tshadow, world, dum))
		L = scale_color(texel_color, material.ka * 1.2);

	
	/*int numLights = sr.w->lights.size();

	for (int j = 0; j < numLights; j++) {
		Vector wi = sr.w->lights[j]->get_direction(sr);
		real ndotwi = sr.normal * wi;



		if (ndotwi > 0.0) {
			bool in_shadow = false;
			if (sr.w->lights[j]->casts_shadows())
			{
				Ray shadowray(sr.local_hit_point + kEpsilon * sr.normal, wi);
				in_shadow = sr.w->lights[j]->in_shadow(shadowray, sr);
			}

			if (!in_shadow)
				L += diffuse_brdf->f(sr, wo, wi) * sr.w->lights[j]->L(sr) * ndotwi;
		}
	}


	if (has_transparency) {
		Ray second_ray(sr.local_hit_point + kEpsilon * sr.ray.d, sr.ray.d);

		real tr = diffuse_brdf->transparency(sr);
		if (tr < 1.0)
			L = tr * L + ((real)1.0 - tr) * sr.w->tracer_ptr->trace_ray(second_ray, sr.depth + 1);
	}*/

	return true;
	
}

__device__ void shade_shadow(world_struct *world, ShadeRec &sr, int seed, float4 &L, float4 &texel_color, bool hitt) {
	if (!hitt) return;
	
	// ==== sun: ================

	material_params material = world->materials[sr.material];


	float ndotwi = -sr.normal * world->light_dir;

	if (ndotwi > 0.f) {
		Ray shadowray;
		shadowray.o = sr.hitPoint + kEPSILON * sr.normal;
		shadowray.d = world->light_dir;
		float t = kHUGEVALUE;
		ShadeRec dummy;
		bool hit = world_hit(shadowray, t, world, dummy);

		if (!hit)
			L = add_colors(L, scale_color(texel_color, material.kd * invPI * world->light_intensity * ndotwi));
	}


}