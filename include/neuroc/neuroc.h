#ifndef NEUROC_H
#define NEUROC_H
#include "matx.h"
#include <stdlib.h>

typedef struct neuroc neuroc;

//// INITIALIZATION:
// Allocate memory for a new neural network.
neuroc *nn_alloc(void);
// Free neural network memory.
void nn_free(neuroc *nn);
// Configure learn hyperparameters.
void nn_config_learn(neuroc *nn, size_t threads, size_t batch_size, double eta);
// Zero-initialized neural network.
void nn_init(neuroc *nn, const size_t n_layers, const size_t *shape);
// Initialize neural netork with random weights and biase.
void nn_init_random(neuroc *nn, const size_t n_layers, const size_t *shape);

//// DISK MANAGEMENT:
// Load weights and biases from a file and create a neural network with these values.
int nn_load(neuroc *nn, const char *file_name);
// Save neural network to a file.
void nn_save(neuroc *nn, const char *file_name);

//// LEARNING:
// Update weights and biases based on the propagated error.
void nn_learn(neuroc *nn, const matx *X, const matx *Y, const size_t train_size);

//// EVALUATION:
// Get prediction from neural network.
const matx *nn_predict(neuroc *nn, const matx *X);

#endif


