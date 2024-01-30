#ifndef SLICE2PLC_H
#define SLICE2PLC_H

#include <cinolib/sliced_object.h>
#include "common.h"

void slice2plc(const SlicedObj<> & obj,
                     Trimesh<>   & plc);

#endif // SLICE2PLC_H
