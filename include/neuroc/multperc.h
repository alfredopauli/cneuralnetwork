#ifndef NEUROC_MULTPERC_H
#define NEUROC_MULTPERC_H
#include "neuroc/matx.h"
#include <stdlib.h>

typedef struct multperc multperc;

//// INITIALIZATION:
  // Allocate memory for a new neural network.
  multperc *ncmp_alloc(
    void
  );
  
  // Free neural network memory.
  void ncmp_free(
    multperc *nn
  );
  
  // Zero-initialized weights and biases neural network.
  void ncmp_init(
    multperc *nn, const size_t n_layers, const size_t *shape
  );
  
  // Initialize neural netork with random weights and biase.
  void ncmp_init_random(
    multperc *nn, const size_t n_layers, const size_t *shape
  );

//// DISK MANAGEMENT:
  // Load weights and biases from a file and create a neural network with these values.
  int ncmp_load(
    multperc *nn, const char *file_name
  );
  
  // Save neural network to a file.
  void ncmp_save(
    multperc *nn, const char *file_name
  );

//// LEARNING:
  // Configure learn hyperparameters.
  void ncmp_config_learn(
    multperc *nn, size_t threads, size_t batch_size, double eta
  );
  
  // Update weights and biases based on the propagated error.
  void ncmp_learn(
    multperc *nn, const matx *X, const matx *Y, const size_t train_size
  );

//// EVALUATION:
  // Get prediction from neural network.
  const matx *ncmp_predict(multperc *nn, const matx *X);

#endif


