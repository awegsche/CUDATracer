#include "MCWorld.h"

#include "NBT/nbttaglist.h"
#include "NBT/nbttagcompound.h"
#include "NBT/nbttagbyte.h"
#include "NBT/nbttagbytearray.h"

#include <iostream>
#include <cuda_runtime.h>

#include "Materials/Materialmanager.h"
#include "Materials/RGBColors.h"
#include "Samplers/sampler.h"

#include "RenderObjects/render_structs.h"


#if !defined WIN64 && !defined WIN32
const QString texturepath = "/home/awegsche/Minecraft/minecraft/textures/blocks/";
#else
const QString texturepath = "G:\\Games\\Minecraft\\res\\minecraft\\textures\\blocks\\";
#endif

#define DEFAULT_KA .3f
#define DEFAULT_KD .8f

MCWorld::MCWorld()
	:num_samples(1)
{
	using namespace std;
	cout << "=================================================\n";
	cout << "= MinecraftRT :: A raytracer for Minecraft      =\n";
	cout << "= Version 0.1                                   =\n";
	cout << "=================================================\n\n";

	cout << "  Current max world size: " << WORLDSIZE_INCHUNKS << " x " << WORLDSIZE_INCHUNKS << " chunks\n";
	cout << "  Block size: " << BLOCKLENGTH << " (if 1.0 world coordinates = game coordinates)\n";
	cout << "  will allocate space for " << WORLDSIZE_INCHUNKS * WORLDSIZE_INCHUNKS * 16 << " chunks\n";

	cout << sizeof(world_struct);

	 
	cudaError_t err = cudaMallocManaged(&device_world, sizeof(world_struct));
	if (err)
		cout << "  device_world allocation failed." << err << "\n";

	int chunks_in_world = WORLDSIZE_INCHUNKS * WORLDSIZE_INCHUNKS * 16;

	if (cudaMallocManaged(&device_world->chunks, chunks_in_world * sizeof(chunk_struct*)))
		cout << "  chunk allocation failed.\n";
	cudaMemset(device_world->chunks, 0, chunks_in_world * sizeof(chunk_struct*));

	device_world->bb_p0.x = -BLOCKLENGTH * CHUNKS_IN_REGION * NREGIONS * CHUNKSIZE;
	device_world->bb_p0.z = -BLOCKLENGTH * CHUNKS_IN_REGION * NREGIONS * CHUNKSIZE;
	device_world->bb_p0.y = .0f;
	device_world->bb_p1.x = BLOCKLENGTH * CHUNKS_IN_REGION * NREGIONS * CHUNKSIZE;
	device_world->bb_p1.z = BLOCKLENGTH * CHUNKS_IN_REGION * NREGIONS * CHUNKSIZE;
	device_world->bb_p1.y = BLOCKLENGTH * 16 * 16;

	cudaMallocManaged(&device_world->blocks, 256 * sizeof(block_struct));


	MaterialManager *m = new MaterialManager();

	for (int i = 1; i < 256; i++)
		_make_block(device_world->blocks[i],
			m->create_matte(.3, .4, m->create_constantcolor(1.0, 0.0, 1.0)), SOLIDBLOCK);
	_make_block(device_world->blocks[0], 0, INVALID); // air 

	_make_block(
		device_world->blocks[BlockInfo::GrassSide],
		m->create_matte(DEFAULT_KA, DEFAULT_KD, m->load_texture(texturepath + "grass_top.png", .0f, 1.0f, .0f)),
		m->create_matte(DEFAULT_KA, DEFAULT_KD, m->load_texture(texturepath + "grass_side.png")),
		SOLIDBLOCK
	);

	_make_block(
		device_world->blocks[BlockInfo::LeavesOak],
		m->create_matte(DEFAULT_KA, DEFAULT_KD, m->load_texture(texturepath + "leaves_oak.png", .0f, 1.0f, .0f), true),
		SOLIDBLOCK
	);
	_make_block(
		device_world->blocks[BlockInfo::Glass],
		m->create_matte(DEFAULT_KA, DEFAULT_KD, m->load_texture(texturepath + "glass.png", .0f, 1.0f, .0f), true),
		SOLIDBLOCK
	);
	_make_block(
		device_world->blocks[BlockInfo::Stone],
		m->create_matte(DEFAULT_KA, DEFAULT_KD, m->load_texture(texturepath + "stone.png")),
		SOLIDBLOCK
	);
	_make_block(
		device_world->blocks[BlockInfo::DoubleStoneSlab],
		m->create_matte(DEFAULT_KA, DEFAULT_KD, m->load_texture(texturepath + "double_stone_slab.png")),
		SOLIDBLOCK
	);
	_make_block(
		device_world->blocks[BlockInfo::CobbleStone],
		m->create_matte(DEFAULT_KA, DEFAULT_KD, m->load_texture(texturepath + "cobblestone.png")),
		SOLIDBLOCK
	);
	_make_block(
		device_world->blocks[BlockInfo::OakWoodPlank], 
		m->create_matte(DEFAULT_KA, DEFAULT_KD, m->load_texture(texturepath + "planks_oak.png")),
		SOLIDBLOCK
	);
	_make_block(
		device_world->blocks[BlockInfo::WaterStill],
		m->create_reflective(DEFAULT_KA, DEFAULT_KD, 1.0f, m->load_texture(texturepath + "water_still.png")),
		SOLIDBLOCK
	);
	_make_block(
		device_world->blocks[BlockInfo::Sand],
		m->create_matte(DEFAULT_KA, 1.2f, m->load_texture(texturepath + "sand.png")),
		SOLIDBLOCK
	);
	_make_block(
		device_world->blocks[BlockInfo::Dirt],
		m->create_matte(DEFAULT_KA, 1.2f, m->load_texture(texturepath + "dirt.png")),
		SOLIDBLOCK
	);
	_make_block(
		device_world->blocks[BlockInfo::Bedrock],
		m->create_matte(DEFAULT_KA, 1.2f, m->load_texture(texturepath + "bedrock.png")),
		SOLIDBLOCK
	);
	_make_block(
		device_world->blocks[BlockInfo::Gravel],
		m->create_matte(DEFAULT_KA, 1.2f, m->load_texture(texturepath + "gravel.png")),
		SOLIDBLOCK
	);

	_make_block(
		device_world->blocks[BlockInfo::LogOak],
		m->create_matte(DEFAULT_KA, DEFAULT_KD, m->load_texture(texturepath + "log_oak_top.png", .0f, 1.0f, .0f)),
		m->create_matte(DEFAULT_KA, DEFAULT_KD, m->load_texture(texturepath + "log_oak.png")),
		SOLIDBLOCK
	);



	m->copy_to_device_memory();

	device_world->texels = m->texels;
	device_world->materials = m->materials;
	device_world->positions = m->positions;
	device_world->dimensions = m->dimensions;

	// Setup light
	device_world->light_dir = _normalize(make_float3(1.0, .5, 3.4442));
	device_world->light_col = rgbcolor(1.0, 1.0, 1.0);
	device_world->light_intensity = 1.5f;


	sampler *S = new sampler();

	S->generate_samples(256);

	device_world->smplr = S->get_device_sampler();
	device_world->num_samples = num_samples;

	set_haze_distance(1.0e6f);
	set_haze_attenuation(0);
	set_haze_strength(0.01f);

	device_world->max_depth = 4;
}


MCWorld::~MCWorld()
{
	if (device_world) {
		for (int i = 0; i < WORLDSIZE_INCHUNKS * WORLDSIZE_INCHUNKS; i++)
			if(device_world->chunks[i])
				cudaFree(device_world->chunks[i]);
		cudaFree(device_world->blocks);
		cudaFree(device_world->dimensions);
		cudaFree(device_world->materials);
		cudaFree(device_world->positions);
		cudaFree(device_world->smplr);
		cudaFree(device_world->texels);
		cudaFree(device_world);
	}
}

void MCWorld::addRegion(const int index, const int X, const int Z, NBTTag *root)
{
	if (root->ID() == NBTTag::TAG_End) return;
	NBTTagList<NBTTagCompound> *regions = static_cast<NBTTagList<NBTTagCompound> *>(root->get_child("Level")->get_child("Sections"));

	int xi = index % CHUNKS_IN_REGION + X * CHUNKS_IN_REGION;
	int zi = index / CHUNKS_IN_REGION + Z * CHUNKS_IN_REGION;


	for (NBTTagCompound* region : regions->_children)
	{
		int Y = ((NBTTagByte*)region->get_child("Y"))->getValue();

		//MCGrid* chunkgrid = new MCGrid();
		//chunkgrid->setup(16, 16, 16, BLOCKLENGTH, Point(X + chunk->x * 16, Y * 16, Z + chunk->y * 16));

		NBTTagByteArray* blocks = ((NBTTagByteArray*)region->get_child("Blocks"));

		device_world->chunks[(xi + NREGIONS * CHUNKS_IN_REGION) + Y * WORLDSIZE_INCHUNKS + (zi + NREGIONS * CHUNKS_IN_REGION)  * WORLDSIZE_INCHUNKS * 16] = _make_chunk(blocks, xi, Y, zi);



	}
	
}

world_struct * MCWorld::get_device_world() const
{
	return device_world;
}

int MCWorld::get_num_samples()
{
	return num_samples;
}

void MCWorld::set_haze_distance(float f)
{
	if (device_world)
		device_world->haze_dist = f;
}

void MCWorld::set_haze_attenuation(int power)
{
	if (device_world)
		device_world->haze_attenuation = power;
}

void MCWorld::set_haze_strength(float f)
{
	if (device_world)
		device_world->haze_strength =  f;
}
