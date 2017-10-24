#pragma once
#include "constants.h"
#include <vector_types.h>
#include "render_structs.h"
#include "MinecraftWorld/MCWorld.h"

//
//extern "C" void render_kernel(
//	uchar4 *dst, const int hres, const int vres, const int num_samples, const float s,
//	float3 eye, float3 u, float3 v, float3 w, float aperture, float d, world_struct* world);

// There will only be one camera class, no inheritance
class Camera
{
private:
	float zoom;
	float d;
	float aperture;

	float3 u, v, w;
	float3 up;
	float3 eye, lookat;

	world_struct* world;

public:
	Camera();
	~Camera();


	void render(uchar4* frame, const int w, const int h) const;

	void expose(uchar4* frame, float4* colors, const int w, const int h, const int sample_count) const;

	void compute_uvw();


	void set_world(world_struct* w);

	void set_eye(float x, float y, float z);
	void move_eye(float x, float y, float z);
	void move_eye_forward(float d);
	void set_lookat(float x, float y, float z);
	void move_eye_left(float d);
	void move_eye_right(float d);
	void move_eye_backward(float d);
	void rotate_up(float d);
	void rotate_down(float d);
	void rotate_left(float d);
	void rotate_right(float d);
	void increase_d();
	void decrease_d();
	void zoom_in();
	void zoom_out();
	void increase_aperture();
	void decrease_aperture();
	
	void set_up(float x, float y, float z);

};

