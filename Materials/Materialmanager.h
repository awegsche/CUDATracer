#pragma once
#include "constants.h"
#include <qstring.h>
#include <vector>
#include <QtGui/qimage.h>

typedef unsigned int texture_pos;
typedef unsigned int material_pos;

enum material_type {
	PHONG	= 0x001,
	MATTE	= 0x002,
	REFL	= 0x003,
	TRANSP	= 0x010
};

struct material_params {
	float ka, kd, kr, exp;
	material_type typ;
	texture_pos position;

};

// Data oriented Material
class MaterialManager
{
public:

	// ======= device pointers ========================

	// the texture informations. Texels, texture dimensions and positions in the stream.
	float4* texels;
	texture_pos* positions;
	uint2* dimensions;

	// the material informations. ambient, diffuse and reflective factors, texture positions
	material_params *materials;

private:
	//======== host memory ============================
	std::vector<float4> host_texels;
	std::vector<uint2> host_dims;
	std::vector<texture_pos> host_positions;
	//texture_pos size;
	texture_pos current_texture;
	std::vector<material_params> host_materials;

public:
	MaterialManager();
	~MaterialManager();

	// create a new texture out of a given png. Out of lazyness one can specify here a multiplicatif color(r, g, b) to avoid
	// dealing with biome colors
	texture_pos load_texture(const QString& filename,
		const float r = 1.f, const float g = 1.f, const float b = 1.f);

	// creates a constant color texture
	texture_pos create_constantcolor(const float r, const float g, const float b);

	// creates a new Matte material
	material_pos create_matte(const float k_ambient, const float k_diffuse, texture_pos position);

	// Copy all the data to device memory, if this step is skipped, nothing will work
	void copy_to_device_memory();
};

