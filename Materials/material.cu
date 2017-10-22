#include "RenderObjects/render_structs.h"
#include <cuda_runtime.h>
#include "Materialmanager.h"
#include "RGBColors.h"

__device__ float4 get_color(float4* texels, texture_pos* positions, uint2* dimensions,
	texture_pos pos, float u, float v) 
{
	uint2 dim = dimensions[pos];

	texture_pos index = positions[pos] + (int)(u * dim.x) + (int)(v * dim.y * dim.x);
	return texels[index];
}

__device__ float4 shade(ShadeRec &sr, material_params* materials,
	float4* texels, texture_pos* positions, uint2* dimensions) {

	//float3 wo = -sr.ray.d;

	material_params material = materials[sr.material];

	float4 L = scale_color(get_color(texels, positions, dimensions, material.position, sr.u, sr.v), material.ka);
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

	return L;
}