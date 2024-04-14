#include "Primitive.h"

std::vector<Primitive*> PrimSet;

int PrimIsInclude (TopoDS_Shape S) {
    for (int i = 0; i < PrimSet.size(); i++)
    {
        if (S.IsEqual(*(PrimSet[i]->Value())))
        {
            return i;
        }
    }
    return -1;
}
