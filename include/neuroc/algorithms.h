#ifndef NEUROC_ALGO_H
#define NEUROC_ALGO_H
#include "neuroc_int.h"

//// ALGORITHMS:
// Propagate the input forward.
void nn_feedforward(neuroc *nn, const matx *input, matx *z, matx *a);
// Propagate the error backwards.
void nn_backpropagate(neuroc *nn, const matx *Y, matx *nabla_w, matx *nabla_b, matxwks *wks, matx *z, matx *a
);

#endif
