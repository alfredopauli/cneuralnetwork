#ifndef __NN_H
#define __NN_H
#include "mat.h"
#include <stdlib.h>

double square_loss(const mat *expected, const mat *layer);

typedef struct {
  size_t size;
  size_t *shape;
  mat *layers;
  mat *z;
  mat *deltas;
  mat *biases;
  mat *weights;
  mat *nabla_b;
  mat *nabla_w;
} neural_network;

void   nn_create(neural_network *nn, const size_t size, const size_t *shape);
void   nn_free(neural_network *nn);
void   nn_randomize(neural_network *nn);
double nn_loss(neural_network *nn, const mat *Y);
void   nn_feedforward(neural_network *nn, const mat *input);
void   nn_backpropagate(neural_network *nn, const mat *expected);
void   nn_update_batch(neural_network *nn, const mat *X, const mat *Y, const size_t batch_size, const double eta);
void   nn_print(neural_network *nn);

#endif

