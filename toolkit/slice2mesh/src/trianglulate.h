#ifndef TRIANGLULATE_H
#define TRIANGLULATE_H

#include <vector>
#include <sys/types.h>

int triangulate_quad(const int                 vids[4],
                     const std::vector<int>  & bot_splits,
                     const std::vector<int>  & top_splits,
                           std::vector<uint> & tris);

#endif // TRIANGLULATE_H
