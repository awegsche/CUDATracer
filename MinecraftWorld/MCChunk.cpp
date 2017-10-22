#include "MCChunk.h"
#include "constants.h"
#include "NBT/nbttaglist.h"
#include "NBT/nbttagcompound.h"
#include "NBT/nbttagbyte.h"
#include "NBT/nbttagbytearray.h"

#include <cuda_runtime.h>
#include "MCWorld.h"

MCChunk::MCChunk(const NBTTagByteArray &blocks)
{

}


MCChunk::~MCChunk()
{
}

chunk_struct* _make_chunk(const NBTTagByteArray * blocks, const int x, const int y, const int z)
{
	chunk_struct* str = new chunk_struct();

	cudaMallocManaged(&str, sizeof(chunk_struct));

	str->p0.x = x * CHUNKSIZE * BLOCKLENGTH;
	str->p1.x = (x + 1) * CHUNKSIZE * BLOCKLENGTH;
	str->p0.y = y * CHUNKSIZE * BLOCKLENGTH;
	str->p1.y = (y + 1) * CHUNKSIZE * BLOCKLENGTH;
	str->p0.z = z * CHUNKSIZE * BLOCKLENGTH;
	str->p1.z = (z + 1) * CHUNKSIZE * BLOCKLENGTH;

	for (int j = 0; j < 16; j++)
		for (int k = 0; k < 16; k++)
			for (int i = 0; i < 16; i++)
			{
				str->blocks[k * CHUNKSTRIDE + j * CHUNKSIZE + i] = (blockid_t)(uchar)blocks->_content[j * CHUNKSTRIDE + k * CHUNKSIZE + i];

				//chunkgrid->addblock(i, j, k, blockid);
			}

	return str;
}
