#ifndef NEUROC_LEARN_H
#define NEUROC_LEARN_H
#include "neuroc_int.h"

typedef struct {
  size_t threads;
  size_t z_size;
  size_t a_size;
  size_t nabla_size;
  size_t temp_size;
  size_t offset;
  double *data;
  pthread_mutex_t lock_sum;
} nn_learning_workspace;
  
void nnlwks_init(nn_learning_workspace *wks, neuroc *nn, size_t threads);
double* nnlwks_get_a(nn_learning_workspace *wks, size_t thread);
double* nnlwks_get_z(nn_learning_workspace *wks, size_t thread);
double* nnlwks_get_nabla_w(nn_learning_workspace *wks, size_t thread);
double* nnlwks_get_nabla_b(nn_learning_workspace *wks, size_t thread);
double* nnlwks_get_sum_nabla_w(nn_learning_workspace *wks, size_t thread);
double* nnlwks_get_sum_nabla_b(nn_learning_workspace *wks, size_t thread);
double* nnlwks_get_temp(nn_learning_workspace *wks, size_t thread);
void nnlwks_lock_sum(nn_learning_workspace *wks);
void nnlwks_unlock_sum(nn_learning_workspace *wks, size_t thread);
void nnlwks_free(nn_learning_workspace *wks);

//// Learning:
// Evaluate nabla_w and nabla_b.
void nn_evaluate_nabla(
  neuroc *nn, 
  const matx *X, const matx *Y, 
  const size_t start, const size_t end
);
// Update weights and biases for one batch.
void nn_update_batch(
  neuroc *nn, 
  const matx *X, const matx *Y, 
  const size_t batch_size, 
  const double eta, 
  const size_t threads,
  matxwks *wks, matx *z, matx *a
);

#endif
