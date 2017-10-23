#pragma once
#include <vector>
#include <vector_types.h>

struct sampler_struct {
	float2 *samples;
	float2 *disk_samples;
	float3 *hemisphere_samples;

	int num_sets;
	int num_samples;

	int count;
	int size;
	int jump;
};

class sampler
{
private:
	int num_sets;
	int num_samples;
	std::vector<float2> samples;
	std::vector<float2> disk_samples;
	std::vector<float3> hemisphere_samples;

public:
	sampler();
	~sampler();

	void generate_samples(const int nums);

	void
		map_samples_to_unit_disk(void);

	void
		map_samples_to_hemisphere(const float p);

	void
		map_samples_to_sphere(void);

	sampler_struct *get_device_sampler();


	// the following functions are not const because they change count and jump

	//float2											// get next sample on unit square
	//	sample_unit_square(void);

	//float2											// get next sample on unit disk
	//	sample_unit_disk(void);

	//float3											// get next sample on unit hemisphere
		//sample_hemisphere(void);

	///<summary>
	/// Samples points onto shphere
	///</summary>
	//float3											// get next sample on unit sphere
	//	sample_sphere(void);
};

float rand_float();
