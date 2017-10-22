#pragma once
#include "Materials/Materialmanager.h"

#define INVALID		0
#define SOLIDBLOCK	1
#define PLANT		2

#define TOP		0x01
#define BOTTOM	0x02
#define SOUTH	0x03
#define NORTH	0x04
#define EAST	0x05
#define WEST	0x06

struct block_struct {
	material_pos material_top;
	material_pos material_side;
	material_pos material_bottom;

	int hittype;
};

class BlockInfo
{
public:
	enum EnBlockId {
		Stone = 1,
		GrassSide = 2,
		Dirt = 3,
		CobbleStone = 4,
		OakWoodPlank = 5,
		WaterFlow = 8,
		WaterStill = 9,
		Sand = 12,
		LogOak = 17,
		LeavesOak = 18,
		Grass = 31,
		Dandelion = 37,
		Poppy = 38,
		FarmLand = 60,
		SugarCanes = 83
	};

public:
	BlockInfo();
	~BlockInfo();
};

inline void _make_block(block_struct &bl, material_pos p, int ht) {
	
	bl.material_top = p;
	bl.material_side = p;
	bl.material_bottom = p;
	bl.hittype = ht;

}

inline void _make_block(block_struct &bl, material_pos p_top, material_pos p_side, int ht) {

	bl.material_top = p_top;
	bl.material_side = p_side;
	bl.material_bottom = p_top;
	bl.hittype = ht;
	
}
