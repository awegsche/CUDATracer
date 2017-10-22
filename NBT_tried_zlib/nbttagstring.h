#ifndef NBTTAGSTRING_H
#define NBTTAGSTRING_H
#include "nbttag.h"
#include <string>

class NBTTagString : public NBTTag
{
public:
    std::string _value;
public:
    NBTTagString();

    // NBTTag interface
public:
    NBTTagID ID() const;
};

#endif // NBTTAGSTRING_H
