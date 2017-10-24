#include "sampler.h"

__device__ float2 sample_disk(sampler_struct* smpl, const int seed) {
	return smpl->disk_samples[(smpl->count++ + seed) % smpl->size];
}

__device__ float2 sample_square(sampler_struct* smpl, const int seed) {
	return smpl->samples[(smpl->count++ + seed) % smpl->size];
}

__device__ float3 sample_hemisphere(sampler_struct* smpl, const int seed) {
	return smpl->hemisphere_samples[(smpl->count++ + seed) % smpl->size];
}