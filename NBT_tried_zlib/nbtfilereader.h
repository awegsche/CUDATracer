#ifndef NBTFILEREADER_H
#define NBTFILEREADER_H

#include "constants.h"
//#include <QString>

#include <iostream>
#include <fstream>
#include <string>

class NBTFileReader
{
private:
    std::string _filename;
    int _X, _Z;

public:
    NBTFileReader(const std::string &filename);

    void Load();

    short readInt16(byte *src, int position);
    int readInt24(byte *src, int position);
    int readInt32(byte *src, int position);
    long readInt64(byte *src, int position);
    long readInt64_BigEndian(byte *src, int position);
};

#endif // NBTFILEREADER_H
