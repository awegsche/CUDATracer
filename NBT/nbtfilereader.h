#ifndef NBTFILEREADER_H
#define NBTFILEREADER_H
#include "constants.h"
#include <QString>
//#include "mcworld.h"
//#include "world.h"

class MCWorld;

class NBTFileReader
{
private:
    QString _filename;
    int _X, _Z;

public:
    NBTFileReader(const QString &folder, const int x, const int z);

    void Load(MCWorld *world);

    qint16 readInt16(byte *src, int position);
    qint32 readInt24(byte *src, int position);
    qint32 readInt32(byte *src, int position);
    qint64 readInt64(byte *src, int position);
    qint64 readInt64_BigEndian(byte *src, int position);
};

#endif // NBTFILEREADER_H
