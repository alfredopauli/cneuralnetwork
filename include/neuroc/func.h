#ifndef NEUROC_FUNC_H
#define NEUROC_FUNC_H
#include "neuroc_int.h"

//// ACTIVATION FUNCTIONS:
// Sigmoid activation function.
void sigmoid(double *dest, const double *src);
// Sigmoid activation derivative.
void dsigmoid(double *dest, const double *src);

//// LOSS FUNCTIONS:
// Square loss function.
double square_loss(const matx *expected, const matx *layer);
// Square loss derivative.
void nabla_square_loss(matx *delta, const matx *layer, const matx *expected);

#endif
