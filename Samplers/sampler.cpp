#include "sampler.h"
#include <vector_functions.h>
#include <random>
#include "constants.h"
#include <cuda_runtime.h>


sampler::sampler()
	: num_sets(83), num_samples(1)
{
}


sampler::~sampler()
{
}

void sampler::generate_samples(const int nums)
{
	num_samples = nums;
	int j = 0;
	for (int p = 0; p < num_sets; p++)
		for (int q = 0; q < nums; q++)
		{
			samples.push_back(make_float2(rand_float(), rand_float()));
		}

	map_samples_to_unit_disk();
	map_samples_to_hemisphere(1.0f);
}

void sampler::map_samples_to_unit_disk(void)
{
	int size = samples.size();
	float r, phi;		// polar coordinates
	float2 sp; 		// sample point on unit disk

	disk_samples.resize(size);


	for (int j = 0; j < size; j++) {
		// map sample point to [-1, 1] X [-1,1]

		sp.x = 2.0f * samples[j].x - 1.0f;
		sp.y = 2.0f * samples[j].y - 1.0f;

		if (sp.x > -sp.y) {			// sectors 1 and 2
			if (sp.x > sp.y) {		// sector 1
				r = sp.x;
				phi = sp.y / sp.x;
			}
			else {					// sector 2
				r = sp.y;
				phi = 2 - sp.x / sp.y;
			}
		}
		else {						// sectors 3 and 4
			if (sp.x < sp.y) {		// sector 3
				r = -sp.x;
				phi = 4 + sp.y / sp.x;
			}
			else {					// sector 4
				r = -sp.y;
				if (sp.y != 0.0)	// avoid division by zero at origin
					phi = 6 - sp.x / sp.y;
				else
					phi = 0.0;
			}
		}

		phi *= PI / 4.0f;

		disk_samples[j].x = r * cos(phi);
		disk_samples[j].y = r * sin(phi);

	}

}

void sampler::map_samples_to_hemisphere(const float p)
{
	int size = samples.size();
	hemisphere_samples.reserve(num_samples * num_sets);

	for (int j = 0; j < size; j++) {
		float cos_phi = cos(2.0 * PI * samples[j].x);
		float sin_phi = sin(2.0 * PI * samples[j].x);
		float cos_theta = pow((1.0 - samples[j].y), 1.0 / (p + 1.0));
		float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
		float pu = sin_theta * cos_phi;
		float pv = sin_theta * sin_phi;
		float pw = cos_theta;
		hemisphere_samples.push_back(make_float3(pu, pv, pw));



	}

}

void sampler::map_samples_to_sphere(void)
{
}

sampler_struct * sampler::get_device_sampler()
{
	sampler_struct *ptr;

	cudaMallocManaged(&ptr, sizeof(sampler_struct));

	ptr->count = 0;
	ptr->jump = 13;
	ptr->size = samples.size();
	ptr->num_sets = num_sets;
	ptr->num_samples = num_samples;

	cudaMalloc(&ptr->samples, sizeof(float2) * samples.size());
	cudaMalloc(&ptr->disk_samples, sizeof(float2) * samples.size());
	cudaMalloc(&ptr->hemisphere_samples, sizeof(float3) * samples.size());

	cudaMemcpy(ptr->samples, samples.data(), sizeof(float2) * samples.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(ptr->disk_samples, disk_samples.data(), sizeof(float2) * samples.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(ptr->hemisphere_samples, hemisphere_samples.data(), sizeof(float3) * samples.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);

	return ptr;
}

//float2 sampler::sample_unit_square(void)
//{
//	return make_float2(0, 0);
//	//return (samples[jump + shuffled_indices[jump + count++ % num_samples]]);
//}

	float rand_float()
	{
		return (float)rand() / RAND_MAX;
	}
