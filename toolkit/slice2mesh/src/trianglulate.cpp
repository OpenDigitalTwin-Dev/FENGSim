#include "trianglulate.h"
#include <assert.h>

enum
{
    _A_ = 0,
    _B_ = 1,
    _C_ = 2,
    _D_ = 3,
};

int triangulate_quad(const int                 vids[4],
                     const std::vector<int>  & bot_splits,
                     const std::vector<int>  & top_splits,
                           std::vector<uint> & tris)
{
    int count = 0;

    assert(vids[_A_] != vids[_B_]);
    assert(vids[_C_] != vids[_D_]);

    std::vector<int> bot_chain;
    copy(bot_splits.begin(), bot_splits.end(), back_inserter(bot_chain));
    bot_chain.push_back(vids[_B_]);

    std::vector<int> top_chain;
    copy(top_splits.begin(), top_splits.end(), back_inserter(top_chain));
    top_chain.push_back(vids[_D_]);

    int pivot = vids[_A_];
    int prev  = vids[_C_];
    for(int curr : top_chain)
    {
        tris.push_back(pivot);
        tris.push_back(curr);
        tris.push_back(prev);
        prev = curr;
        ++count;
    }

    pivot = vids[_D_];
    prev  = vids[_A_];
    for(int curr : bot_chain)
    {
        tris.push_back(pivot);
        tris.push_back(prev);
        tris.push_back(curr);
        prev = curr;
        ++count;
    }

    return count;
}
