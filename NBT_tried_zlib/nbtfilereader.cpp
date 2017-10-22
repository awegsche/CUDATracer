#include "nbtfilereader.h"
#include "nbttag.h"
#include "bigendianreader.h"
#include "chunk.h"
//#include "world.h"

#include "constants.h"
#include "zlib.h"
//#include <QDebug>
//#include <QFile>
//#include "QDataStream"

#define CHUNK 16384
#define OUT_CHUNK 32768

typedef unsigned int uint;

NBTFileReader::NBTFileReader(const std::string &filename)
    :_filename(filename)
{
    //normally: get x and z from filename
    //but now I load only the file -1.0, so

    _X = -1;
    _Z = 0;

//    FILE* file = std::fopen(filename.toStdString().c_str(), "rb");
//    char* buffer = new char[file->_bufsiz];
//    fgets(buffer, file->_bufsiz, file);

//    std::fclose(file);




}

void NBTFileReader::Load()
{
	std::ifstream file(_filename, std::ios::binary);
   // file.open(QIODevice::ReadOnly);
    //byte* buffer = new byte[1024];
	if (!file.is_open()) {
		std::cout << "error loading " << _filename;
		return;
	}
	char* buffer = new char[4096];

	file.read(buffer, 4096);

    BigEndianReader Lreader((byte*)buffer);
    //QDataStream stream(&file);

    // OK, nun kommt das NBT file format:
    // die ersten 1024 * 4 byte sind Chunk-offset informationen

	int ret, flush;
	unsigned have;
	z_stream strm;

    for(int i = 0; i< 1024; i++){




       int offset = Lreader.readInt24();
       Lreader.readByte();
       if (offset == 0) continue;
       //stream.skipRawData(1);

//        qDebug() << "\noffset["<<i<<"] = "<<offset<<"";

        Chunk* c = new Chunk();


        file.seekg(offset*4096, std::ios::beg);


        byte* tmp = new byte[4];
        file.read((char*)tmp, 4);

        int length = readInt32(tmp, 0);

        file.seekg(offset * 4096 + 5, std::ios::beg);

		int chunk_length = length + 5;

		byte* chunkdata = new byte[chunk_length];
        auto size = BigEndianReader::toByteArray(length);
		memcpy(chunkdata, size, 4);

		char* _chunkdata = new char[length + 1];
		file.read(_chunkdata, length + 1);
		memcpy(chunkdata + 4, _chunkdata, length + 1);

		byte* out = new byte[CHUNK];
		
		byte* dest = new byte[OUT_CHUNK];


       // auto chunkdata = size.append(_chunkdata);

		/* allocate inflate state */
		strm.zalloc = Z_NULL;
		strm.zfree = Z_NULL;
		strm.opaque = Z_NULL;
		strm.avail_in = 0;
		strm.next_in = Z_NULL;
		ret = inflateInit(&strm);
		if (ret != Z_OK)
			std::cout << "inflate error";

		/* decompress until deflate stream ends or end of file */
		do {


			strm.avail_in = chunk_length;
			
			
			strm.next_in = chunkdata;

			int pos = 0;

			/* run inflate() on input until output buffer not full */
			do {
				strm.avail_out = CHUNK;
				strm.next_out = out;
				ret = inflate(&strm, Z_NO_FLUSH);
				
				switch (ret) {
				case Z_NEED_DICT:
					ret = Z_DATA_ERROR;     /* and fall through */
				case Z_DATA_ERROR:
				case Z_MEM_ERROR:
					(void)inflateEnd(&strm);
					std::cout << "MEM_ERROR";
					break;
				}
				have = CHUNK - strm.avail_out;
				memcpy(dest + pos, out, have);
				pos += have;
				
			} while (strm.avail_out == 0);
			/* done when inflate() says it's done */
		} while (ret != Z_STREAM_END);
		/* clean up and return */
		(void)inflateEnd(&strm);
		
	
       
        BigEndianReader R(dest);

        NBTTag *root = fromFile(R);

        //world->addChunk(i%32, i/32, root);
       // w->add_chunk(root, _X*32 + i%32, _Z*32 + i/32);


    }

    file.close();

}

long NBTFileReader::readInt64(byte *src, int position)
{
	long i = 0;
    i |= src[position] << 56;
    i |= src[position + 1] << 48;
    i |= src[position + 2] << 40;
    i |= src[position + 3] << 32;
    i |= src[position + 4] << 24;
    i |= src[position + 5] << 16;
    i |= src[position + 6] << 8;
    i |= src[position + 7];

    return i;
}

long NBTFileReader::readInt64_BigEndian(byte *src, int position)
{
	long i = 0;
    i |= src[position + 7] << 56;
    i |= src[position + 6] << 48;
    i |= src[position + 5] << 40;
    i |= src[position + 4] << 32;
    i |= src[position + 3] << 24;
    i |= src[position + 2] << 16;
    i |= src[position + 1] << 8;
    i |= src[position];

    return i;
}

short NBTFileReader::readInt16(byte *src, int position)
{
	short i = 0;

    i |= src[position] << 8;
    i |= src[position + 1];

    return i;
}
int NBTFileReader::readInt24(byte *src, int position)
{
    int i = 0;
    i |= (int)src[position] << 16;
    i |= (int)src[position + 1] << 8;
    i |= (int)src[position + 2];

    return i;
}
int NBTFileReader::readInt32(byte *src, int position)
{
    int i = 0;
    i |= (uint)src[position] << 24;
    i |= (uint)src[position + 1] << 16;
    i |= (uint)src[position + 2] << 8;
    i |= (uint)src[position + 3];

    return i;
}

