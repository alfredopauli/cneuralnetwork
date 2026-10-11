#ifndef NEUROC_INT_H
#define NEUROC_INT_H
#include "matx.h"
#include "neuroc.h"
#include <stdlib.h>

// Helper functions:
static inline size_t min(size_t a, size_t b)
{
  return (a < b) ? (a) : (b);
}

// "neuroc" definition:
struct neuroc {
  size_t n_layers;
  size_t *shape;
  matx *weights;
  matx *biases;
  matx pred;
};

#endif
