#pragma once
#include "NBT/nbttag.h"
#include "MCChunk.h"
#include <vector>
#include "Materials/Materialmanager.h"
#include "BlockInfo.h"

// The side length of one block
// If 1.0, ray tracer world coordinates and game coordinates are identical.
#define BLOCKLENGTH 1.0f

// The regions with X,Z \in [-NREGIONS, ... , NREGIONS] are loaded.
#define NREGIONS 2
#define WORLDSIZE 4 // in regions, 2 * NREGIONS 
#define CHUNKS_IN_REGION 32
#define WORLDSIZE_INCHUNKS 128 // in chunks, CHUNKS_IN_REGION * WORLDSIZE

struct world_struct {

	// 1. chunks
	chunk_struct** chunks;
	float3 bb_p0, bb_p1;

	// 2. materials and textures
	float4* texels;
	texture_pos* positions;
	uint2* dimensions;
	material_params *materials;

	// 3. block info
	block_struct* blocks;

	// 4. lights
	float3 light_dir;
	float4 light_col;
	float light_intensity;

};

// Data oriented Minecraft World
class MCWorld
{
	// ====== device memory: ======================
	world_struct *device_world;

	// ====== host memory: ========================
	std::vector<MCChunk> host_chunks;

public:
	MCWorld();
	~MCWorld();

public:
	void addRegion(const int index, const int X, const int Z, NBTTag *root);

	world_struct* get_device_world() const;
};

