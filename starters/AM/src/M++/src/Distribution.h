// file: Distribution.h
// author: Christian Wieners
// $Header: /public/M++/src/Distribution.h,v 1.1.1.1 2007-02-19 15:55:20 wieners Exp $

#ifndef _DISTRIBUTION_H_
#define _DISTRIBUTION_H_

#include "Mesh.h"

void Distribute (Mesh&, const string&);
void DistributeOverlap (Mesh&, const string&);

#endif
