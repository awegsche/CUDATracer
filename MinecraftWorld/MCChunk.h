#pragma once
#include "NBT/nbttag.h"
#include "NBT/nbttagbytearray.h"
#include "constants.h"
#include <vector>
#include <vector_types.h>

// how long is a chunk
#define CHUNKSIZE 16

// blocks in one chunk level
#define CHUNKSTRIDE 256

// blocks in a chunk
#define CHUNKCELLS 4096

typedef unsigned short blockid_t;

struct chunk_struct {
	blockid_t blocks[CHUNKCELLS];
	float3 p0, p1;
};

chunk_struct* _make_chunk(const NBTTagByteArray *blocks, const int x, const int y, const int z);

class MCChunk
{
	// host memory
	blockid_t blockids[CHUNKCELLS];


public:
	MCChunk(const NBTTagByteArray &blocks);
	~MCChunk();
};

