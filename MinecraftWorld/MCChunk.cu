#ifndef MCCHUNK_CU
#define MCCHUNK_CU

#include "RenderObjects/render_structs.h"
#include <cuda_runtime.h>
#include "MCChunk.h"
#include "BlockInfo.h"
#include "Block.cu"


__device__ bool hit_mcchunk(chunk_struct *chunk, Ray &ray, float& tmin, ShadeRec &sr, block_struct *blocks) {
	//Material* mat_ptr = sr.material_ptr;
	float ox = ray.o.x;
	float oy = ray.o.y;
	float oz = ray.o.z;
	float dx = ray.d.x;
	float dy = ray.d.y;
	float dz = ray.d.z;
	float x0 = chunk->p0.x;
	float y0 = chunk->p0.y;
	float z0 = chunk->p0.z;
	float x1 = chunk->p1.x;
	float y1 = chunk->p1.y;
	float z1 = chunk->p1.z;
	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;
	// the following code includes modifications from Shirley and Morley (2003)

	float a = 1.0 / dx;
	if (a >= 0) {
		tx_min = (x0 - ox) * a;
		tx_max = (x1 - ox) * a;
	}
	else {
		tx_min = (x1 - ox) * a;
		tx_max = (x0 - ox) * a;
	}

	float b = 1.0 / dy;
	if (b >= 0) {
		ty_min = (y0 - oy) * b;
		ty_max = (y1 - oy) * b;
	}
	else {
		ty_min = (y1 - oy) * b;
		ty_max = (y0 - oy) * b;
	}

	float c = 1.0 / dz;
	if (c >= 0) {
		tz_min = (z0 - oz) * c;
		tz_max = (z1 - oz) * c;
	}
	else {
		tz_min = (z1 - oz) * c;
		tz_max = (z0 - oz) * c;
	}

	float t0, t1;

	if (tx_min > ty_min)
		t0 = tx_min;
	else
		t0 = ty_min;

	if (tz_min > t0)
		t0 = tz_min;

	if (tx_max < ty_max)
		t1 = tx_max;
	else
		t1 = ty_max;

	if (tz_max < t1)
		t1 = tz_max;

	if (t0 > t1)
		return false;


	// initial cell coordinates

	int ix, iy, iz;

	
	if (_inside_bb(ray.o, chunk->p0, chunk->p1)) {  			// does the ray start inside the grid?
		ix = clamp((ox - x0) * CHUNKSIZE / (x1 - x0), 0, CHUNKSIZE - 1);
		iy = clamp((oy - y0) * CHUNKSIZE / (y1 - y0), 0, CHUNKSIZE - 1);
		iz = clamp((oz - z0) * CHUNKSIZE / (z1 - z0), 0, CHUNKSIZE - 1);
	}
	else {
		float3 p = ray.o + ray.d * t0;  // initial hit point with grid's bounding box
		ix = clamp((p.x - x0) * CHUNKSIZE / (x1 - x0), 0, CHUNKSIZE - 1);
		iy = clamp((p.y - y0) * CHUNKSIZE / (y1 - y0), 0, CHUNKSIZE - 1);
		iz = clamp((p.z - z0) * CHUNKSIZE / (z1 - z0), 0, CHUNKSIZE - 1);
	}

	

	// ray parameter increments per cell in the x, y, and z directions

	float dtx = (tx_max - tx_min) / CHUNKSIZE;
	float dty = (ty_max - ty_min) / CHUNKSIZE;
	float dtz = (tz_max - tz_min) / CHUNKSIZE;

	float 	tx_next, ty_next, tz_next;
	int 	ix_step, iy_step, iz_step;
	int 	ix_stop, iy_stop, iz_stop;

	if (dx > 0) {
		tx_next = tx_min + (ix + 1) * dtx;
		ix_step = +1;
		ix_stop = CHUNKSIZE;
	}
	else {
		tx_next = tx_min + (CHUNKSIZE - ix) * dtx;
		ix_step = -1;
		ix_stop = -1;
	}

	if (dx == 0.0) {
		tx_next = kHUGEVALUE;
		ix_step = -1;
		ix_stop = -1;
	}


	if (dy > 0) {
		ty_next = ty_min + (iy + 1) * dty;
		iy_step = +1;
		iy_stop = CHUNKSIZE;
	}
	else {
		ty_next = ty_min + (CHUNKSIZE - iy) * dty;
		iy_step = -1;
		iy_stop = -1;
	}

	if (dy == 0.0) {
		ty_next = kHUGEVALUE;
		iy_step = -1;
		iy_stop = -1;
	}

	if (dz > 0) {
		tz_next = tz_min + (iz + 1) * dtz;
		iz_step = +1;
		iz_stop = CHUNKSIZE;
	}
	else {
		tz_next = tz_min + (CHUNKSIZE - iz) * dtz;
		iz_step = -1;
		iz_stop = -1;
	}

	if (dz == 0.0) {
		tz_next = kHUGEVALUE;
		iz_step = -1;
		iz_stop = -1;
	}

	/*if (tx_next < 0) tx_next = kHUGEVALUE;
	if (ty_next < 0) ty_next = kHUGEVALUE;
	if (tz_next < 0) tz_next = kHUGEVALUE;*/

	// Test if there is a block face glued to the bounding box:

	uint block_ptr = chunk->blocks[ix + CHUNKSIZE * iy + CHUNKSTRIDE * iz];
	float3 block_p0 = make_float3(x0 + CHUNKSIZE * BLOCKLENGTH, y0 + CHUNKSIZE * BLOCKLENGTH, z0 + CHUNKSIZE * BLOCKLENGTH);
	if (block_ptr) {
		float t_before = kHUGEVALUE;

		float tx_min_pp = tx_next - dtx;
		float ty_min_pp = ty_next - dty;
		float tz_min_pp = tz_next - dtz;

		if (ix != 0 && ix != (CHUNKSIZE - 1)) tx_min_pp = -kHUGEVALUE;
		if (iy != 0 && iy != (CHUNKSIZE - 1)) ty_min_pp = -kHUGEVALUE;
		if (iz != 0 && iz != (CHUNKSIZE - 1)) tz_min_pp = -kHUGEVALUE;


		if (tx_min_pp > ty_min_pp && tx_min_pp > tz_min_pp) {
			sr.normal = make_float3(-(float)ix_step, 0, 0);
			sr.hdir = ix_step > 0 ? SOUTH : NORTH;
			t_before = tx_min_pp;
		}
		else if (ty_min_pp > tz_min_pp) {
			sr.normal = make_float3(0, -(float)iy_step, 0);
			sr.hdir = iy_step > 0 ? BOTTOM : TOP;
			t_before = ty_min_pp;

		}
		else {
			sr.normal = make_float3(0, 0, -(float)iz_step);
			sr.hdir = iz_step > 0 ? WEST : EAST;
			t_before = tz_min_pp;

		}
		if (block_ptr && blockhit(ray, sr, blocks, block_ptr, t_before)) {
			tmin = t_before;


			return (true);
		}
	}


	// traverse the grid
	//t = kHugeValue;
	float t_before = kHUGEVALUE;

	while (true) {
		if (tx_next < ty_next && tx_next < tz_next) {
			sr.normal = make_float3(-(float)ix_step, 0, 0);
			sr.hdir = ix_step > 0 ? SOUTH : NORTH;
			t_before = tx_next;
			tx_next += dtx;
			ix += ix_step;
			if (ix == ix_stop) {
				
				return (false);
			}

			uint block_ptr = chunk->blocks[ix + CHUNKSIZE * iy + CHUNKSTRIDE * iz];
			float3 block_p0 = make_float3(x0 + CHUNKSIZE * BLOCKLENGTH, y0 + CHUNKSIZE * BLOCKLENGTH, z0 + CHUNKSIZE * BLOCKLENGTH);

			if (block_ptr && blockhit(ray, sr, blocks, block_ptr, t_before)/* && tmin < tx_next*/) {
				tmin = t_before;
				return true;
			}
		}
		else {
			if (ty_next < tz_next) {
				sr.normal = make_float3(0.0, -(float)iy_step, 0);
				sr.hdir = iy_step > 0 ? BOTTOM : TOP;
				
				t_before = ty_next;
				ty_next += dty;
				iy += iy_step;
				if (iy == iy_stop) {
					
					return (false);
				}
				uint block_ptr = chunk->blocks[ix + CHUNKSIZE * iy + CHUNKSTRIDE * iz];
				float3 block_p0 = make_float3(x0 + CHUNKSIZE * BLOCKLENGTH, y0 + CHUNKSIZE * BLOCKLENGTH, z0 + CHUNKSIZE * BLOCKLENGTH);

				if (block_ptr && blockhit(ray, sr, blocks, block_ptr, t_before)/* && tmin < ty_next*/) {
					//material_ptr = object_ptr->get_material();
					tmin  = t_before;
					
					return true;
				}
				
			}
			else {
				sr.normal = make_float3(0.0, 0.0, -(float)iz_step);
				sr.hdir = iz_step > 0 ? WEST : EAST;
				
				t_before = tz_next;
				tz_next += dtz;
				iz += iz_step;
				if (iz == iz_stop) {
					
					return (false);
				}
				uint block_ptr = chunk->blocks[ix + CHUNKSIZE * iy + CHUNKSTRIDE * iz];
				float3 block_p0 = make_float3(x0 + CHUNKSIZE * BLOCKLENGTH, y0 + CHUNKSIZE * BLOCKLENGTH, z0 + CHUNKSIZE * BLOCKLENGTH);

				if (block_ptr && blockhit(ray, sr, blocks, block_ptr, t_before) /*&& tmin < tz_next*/) {
					tmin  = t_before;
					
					
					return true;
				}
			}
		}
	}
}

#endif