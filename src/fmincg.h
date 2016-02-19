#ifndef _FMINCG_H
#define _FMINCG_H

#include "NNDefs.h"
// Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13

/* Ported to C++ by Ricardo Pieper. Date 2016-02-03 */

// a bunch of constants for line searches:
// RHO and SIG are the constants in the Wolfe-Powell conditions
const double RHO = 0.01;
const double SIG = 0.5;

// don't reevaluate within 0.1 of the limit of the current bracket
const double INT = 0.1;
const double EXT = 3.0;   // extrapolate maximum 3 times the current bracket
const double MAX = 20;    // max 20 function evaluations per line search

// maximum allowed slope ratio
const double RATIO = 100;


Mtx fmincg(const std::function<Cost (Mtx)> f,
            Mtx X,
            const unsigned int maxIters);





#endif  // _FMINCG_H
