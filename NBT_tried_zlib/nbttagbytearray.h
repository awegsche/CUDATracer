#ifndef NBTTAGBYTEARRAY_H
#define NBTTAGBYTEARRAY_H
#include "nbttag.h"
#include "constants.h"
//#include "QByteArray"

class NBTTagByteArray : public NBTTag
{
public:
    byte* _content;

public:
    NBTTagByteArray();


    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGBYTEARRAY_H
