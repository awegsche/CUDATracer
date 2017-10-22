#include "BlockInfo.h"
#include "constants.h"
#include "MCWorld.h"

#include "RenderObjects/render_structs.h"
#include <cuda_runtime.h>

__device__ bool blockhit(Ray ray, ShadeRec &sr, block_struct *blocks, uint blockid, float &t) {
	block_struct B = blocks[blockid];


	switch (B.hittype)
	{
	case SOLIDBLOCK:
		sr.hitPoint = ray.o + ray.d * t;

		switch (sr.hdir) {
		case TOP:
			sr.u = fmod(sr.hitPoint.x, BLOCKLENGTH);
			sr.v = fmod(sr.hitPoint.z, BLOCKLENGTH);
			sr.material = B.material_top;
			break;
		case BOTTOM:
			sr.u = fmod(sr.hitPoint.x, BLOCKLENGTH);
			sr.v = fmod(sr.hitPoint.z, BLOCKLENGTH);
			sr.material = B.material_bottom;
			break;

		case EAST:

			sr.u = fmod(sr.hitPoint.x, BLOCKLENGTH);
			sr.v = fmod(sr.hitPoint.y, BLOCKLENGTH);
			sr.material = B.material_side;
			break;

		case WEST:
			sr.u = fmod(sr.hitPoint.x, BLOCKLENGTH);
			sr.v = fmod(sr.hitPoint.y, BLOCKLENGTH);
			sr.material = B.material_side;
			break;

		case NORTH:
			sr.u = fmod(sr.hitPoint.z, BLOCKLENGTH);
			sr.v = fmod(sr.hitPoint.y, BLOCKLENGTH);
			sr.material = B.material_side;
			break;

		case SOUTH:
			sr.u = fmod(sr.hitPoint.z, BLOCKLENGTH);
			sr.v = fmod(sr.hitPoint.y, BLOCKLENGTH);
			sr.material = B.material_side;
			break;
		}

		if (sr.u < 0) sr.u = -sr.u;
		if (sr.v < 0) sr.v = -sr.v;
		sr.v = 1.0 - sr.v;

		sr.ray.o = ray.o;
		sr.ray.d = ray.d;

		return true;
	}

	return false;
}