#include "RenderObjects/render_structs.h"
#include <cuda_runtime.h>
#include "Materialmanager.h"
#include "RGBColors.h"
#include "MinecraftWorld/MCWorld.cu"
#include "Samplers/sampler.h"

// forward definition of trac_ray
__device__ rgbcol trace_ray(
	Ray &ray, world_struct *world, const int seed, int depth);

__device__ rgbacol get_color(rgbacol* texels, texture_pos* positions, uint2* dimensions,
	texture_pos pos, float u, float v) 
{
	uint2 dim = dimensions[pos];
	int iv = u * (dim.x-1);
	int iu = v * (dim.y-1);

	texture_pos index = positions[pos] + iu + iv * dim.x;
	return texels[index];
}

__device__ rgbcol shade(ShadeRec &sr, world_struct *world, const float3 &sp, bool &hitt, rgbacol &texel_color) {

	if (!hitt) return;

	//float3 wo = -sr.ray.d;

	material_params material = world->materials[sr.material];
	texel_color = get_color(world->texels, world->positions, world->dimensions, material.position, sr.u, sr.v);

	if ((material.typ & TRANSP) && texel_color.w < 1.0f)
	{
		hitt = false;
		return rgbcolor(0.0f);
	}
	hitt = true;

	// ==== Simple Ambient ======
	// lambertian rho
	rgbcol L = rgbcolor(texel_color) * material.ka * 0.2f;
	float3 u, v, w;

	w = sr.normal;
	v = _normalize(w ^ make_float3(-0.0073f, 1.0f, 0.0034f));
	u = v ^ w;
	//float3 sp = sample_hemisphere(world->smplr, seed);

	Ray shadowray;
	shadowray.o = sr.hitPoint() + kEPSILON * sr.normal;
	shadowray.d = sp.x * u + sp.y * v + sp.z * w;
	ShadeRec dum;
	float tshadow = kHUGEVALUE;
	if (!world_hit(shadowray, tshadow, world, dum))
		L = L + rgbcolor(texel_color) * material.ka;

	//L = rgbcolor(sr.u, sr.v, 0.0f);

	return L;
	
}

__device__ void shade_shadow(world_struct *world, ShadeRec &sr, const float3 &sp, rgbcol &L, rgbacol &texel_color, bool hitt) {
	if (!hitt) return;
	
	// ==== sun: ================



	float ndotwi = -sr.normal * world->light_dir;

	if (ndotwi > 0.f) {
		Ray shadowray;
		shadowray.o = sr.hitPoint() + kEPSILON * sr.normal;
		shadowray.d = world->light_dir;
		float t = kHUGEVALUE;
		ShadeRec dummy;
		bool hit = world_hit(shadowray, t, world, dummy);

		if (!hit) {
			material_params material = world->materials[sr.material];
			L = add_colors(L, scale_color(rgbcolor(texel_color), material.kd * invPI * world->light_intensity * ndotwi));
		}
	}


}

__device__ bool shade_reflection(world_struct *world, ShadeRec &sr, float3 &wo, float3 &wi, rgbcol &L, int seed, int depth) {

	material_params material = world->materials[sr.material];

	if (!material.typ & REFL) return false;

	float ndotwo = sr.normal * wo;
	wi = -wo + 2.0 * sr.normal * ndotwo;


	Ray reflected_ray;
	reflected_ray.d = wi;
	reflected_ray.o = sr.hitPoint() + kEPSILON * sr.normal;

	L = add_colors(L, scale_color(trace_ray(reflected_ray, world, seed, depth + 1), material.kr / (sr.normal * wi)));

	return true;
}