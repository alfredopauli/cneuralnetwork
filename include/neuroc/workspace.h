#ifndef NEUROC_WORKSPACE_H
#define NEUROC_WORKSPACE_H
#include "neuroc/multperc.h"
#include <stdlib.h>

// Learning workspace
typedef struct {
  size_t threads;
  size_t z_size;
  size_t a_size;
  size_t nabla_size;
  size_t temp_size;
  size_t offset;
  double *data;
  pthread_mutex_t lock_sum;
} multperc_wks;

void mpwks_init(multperc_wks *wks, neuroc_multperc *nn, size_t threads)
{
  wks->threads = threads;
  wks->z_size     = 0;
  wks->a_size     = 0;
  wks->nabla_size = 0;
  size_t max_size = 0;
  size_t l, size;
  for (l=0; l < nn->n_layers; l++)
  {
    wks->a_size += nn->shape[l];
    if (l > 0)
    {
      size = nn->shape[l-1] * nn->shape[l];
      if (size > max_size)
        max_size = size;
      wks->z_size += nn->shape[l];
      wks->nabla_size += nn->shape[l];
    }
  }
  wks->temp_size = 3 * max_size;
  wks->offset = wks->a_size + wks->z_size + 4 * wks->nabla_size + wks->temp_size;
  wks->data = (double *)malloc(sizeof(double) * threads * wks->offset);
  pthread_mutex_init(&wks->lock_sum, NULL);
}

double* mpwks_get_a(multperc_wks *wks, size_t thread)
{
  return wks->data+(wks->offset*thread);
}

double* mpwks_get_z(multperc_wks *wks, size_t thread)
{
  return wks->data+(wks->offset*thread)
      + wks->a_size;
}

double* mpwks_get_nabla_w(multperc_wks *wks, size_t thread)
{
  return wks->data+(wks->offset*thread)
      + wks->a_size
      + wks->z_size;
}

double* mpwks_get_nabla_b(multperc_wks *wks, size_t thread)
{
  return wks->data+(wks->offset*thread)
      + wks->a_size
      + wks->z_size
      + wks->nabla_size;
}

double* mpwks_get_sum_nabla_w(multperc_wks *wks, size_t thread)
{
  return wks->data+(wks->offset*thread)
      + wks->a_size
      + wks->z_size
      + 2*wks->nabla_size;
}

double* mpwks_get_sum_nabla_b(multperc_wks *wks, size_t thread)
{
  return wks->data+(wks->offset*thread)
      + wks->a_size
      + wks->z_size
      + 3*wks->nabla_size;
}

double* mpwks_get_temp(multperc_wks *wks, size_t thread)
{
  return wks->data+(wks->offset*thread)
      + wks->a_size
      + wks->z_size
      + 4*wks->nabla_size;
}

void mpwks_lock_sum(multperc_wks *wks)
{
  pthread_mutex_lock(&wks->lock_sum);
}

void mpwks_unlock_sum(multperc_wks *wks, size_t thread)
{
  pthread_mutex_unlock(&wks->lock_sum);
}

void mpwks_free(multperc_wks *wks)
{
  free(wks->data);
  pthread_mutex_destroy(&wks->lock_sum);
}

#endif

