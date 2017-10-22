#ifndef MCWORLD_CU
#define MCWORLD_CU

#include "MCWorld.h"
#include "RenderObjects/render_structs.h"
#include <cuda_runtime.h>
#include "MCChunk.cu"

__device__ bool world_hit(
	Ray &ray, float &t,
	const float3 &p0, const float3 &p1, const int nx, const int ny, const int nz,
	chunk_struct** cells, ShadeRec &sr, block_struct *blocks) 
{
	//Material* mat_ptr = sr.material_ptr;
	t = kHUGEVALUE;

	float ox = ray.o.x;
	float oy = ray.o.y;
	float oz = ray.o.z;
	float dx = ray.d.x;
	float dy = ray.d.y;
	float dz = ray.d.z;
	float x0 = p0.x;
	float y0 = p0.y;
	float z0 = p0.z;
	float x1 = p1.x;
	float y1 = p1.y;
	float z1 = p1.z;
	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;
	// the following code includes modifications from Shirley and Morley (2003)

	double a = 1.0 / dx;
	if (a >= 0) {
		tx_min = (x0 - ox) * a;
		tx_max = (x1 - ox) * a;
	}
	else {
		tx_min = (x1 - ox) * a;
		tx_max = (x0 - ox) * a;
	}

	double b = 1.0 / dy;
	if (b >= 0) {
		ty_min = (y0 - oy) * b;
		ty_max = (y1 - oy) * b;
	}
	else {
		ty_min = (y1 - oy) * b;
		ty_max = (y0 - oy) * b;
	}

	double c = 1.0 / dz;
	if (c >= 0) {
		tz_min = (z0 - oz) * c;
		tz_max = (z1 - oz) * c;
	}
	else {
		tz_min = (z1 - oz) * c;
		tz_max = (z0 - oz) * c;
	}

	double t0, t1;

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
		return(false);


	// initial cell coordinates

	int ix, iy, iz;

	if (_inside_bb(ray.o, p0, p1)) {  			// does the ray start inside the grid?
		ix = clamp((ox - x0) * nx / (x1 - x0), 0, nx - 1);
		iy = clamp((oy - y0) * ny / (y1 - y0), 0, ny - 1);
		iz = clamp((oz - z0) * nz / (z1 - z0), 0, nz - 1);
	}
	else {
		float3 p = ray.o + t0 * ray.d;  // initial hit point with grid's bounding box
		ix = clamp((p.x - x0) * nx / (x1 - x0), 0, nx - 1);
		iy = clamp((p.y - y0) * ny / (y1 - y0), 0, ny - 1);
		iz = clamp((p.z - z0) * nz / (z1 - z0), 0, nz - 1);
	}

	// ray parameter increments per cell in the x, y, and z directions

	double dtx = (tx_max - tx_min) / nx;
	double dty = (ty_max - ty_min) / ny;
	double dtz = (tz_max - tz_min) / nz;

	double 	tx_next, ty_next, tz_next;
	int 	ix_step, iy_step, iz_step;
	int 	ix_stop, iy_stop, iz_stop;

	if (dx > 0) {
		tx_next = tx_min + (ix + 1) * dtx;
		ix_step = +1;
		ix_stop = nx;
	}
	else {
		tx_next = tx_min + (nx - ix) * dtx;
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
		iy_stop = ny;
	}
	else {
		ty_next = ty_min + (ny - iy) * dty;
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
		iz_stop = nz;
	}
	else {
		tz_next = tz_min + (nz - iz) * dtz;
		iz_step = -1;
		iz_stop = -1;
	}

	if (dz == 0.0) {
		tz_next = kHUGEVALUE;
		iz_step = -1;
		iz_stop = -1;
	}

	if (tx_next < 0) tx_next = kHUGEVALUE;
	if (ty_next < 0) ty_next = kHUGEVALUE;
	if (tz_next < 0) tz_next = kHUGEVALUE;
	// traverse the grid
	//t = kHugeValue;
	//real t_before = kHugeValue;

	while (true) {
		chunk_struct *block_ptr = cells[ix + nx * iy + nx * ny * iz];
		if (tx_next < ty_next && tx_next < tz_next) {
			//real tmin = tx_next - kEpsilon;
			//Material* mptr = sr.material_ptr;
			if (block_ptr && hit_mcchunk(block_ptr, ray, t, sr, blocks) && t < tx_next) {

				return true;
			}
			//sr.material_ptr = mptr;
			tx_next += dtx;
			ix += ix_step;

			if (ix == ix_stop) {
				//sr.material_ptr = mat_ptr;
				return false;
			}
		}
		else {
			if (ty_next < tz_next) {
				//Material* mptr = sr.material_ptr;
				//real tmin = ty_next - kEpsilon;
				if (block_ptr && hit_mcchunk(block_ptr, ray, t, sr, blocks) && t < ty_next) {
					//material_ptr = object_ptr->get_material();

					return true;
				}
				//sr.material_ptr = mptr;
				ty_next += dty;
				iy += iy_step;
				//mat_ptr

				if (iy == iy_stop) {
					//sr.material_ptr = mat_ptr;
					return false;
				}
			}
			else {
				//Material* mptr = sr.material_ptr;
				//real tmin = tz_next - kEpsilon;
				//material_ptr = sr.material_ptr;
				if (block_ptr && hit_mcchunk(block_ptr, ray, t, sr, blocks) && t < tz_next) {

					return true;
				}
				//sr.material_ptr = mptr;
				tz_next += dtz;
				iz += iz_step;

				if (iz == iz_stop) {
					//sr.material_ptr = mat_ptr;
					return false;
				}
			}
		}
	}
}

#endif