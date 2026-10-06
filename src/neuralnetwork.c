#include "neuralnetwork.h"
#include "mat.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


void sigmoid(double *dest, const double *src)
{
  (*dest) = 1 / (1 + exp(-(*src)));
}

void dsigmoid(double *dest, const double *src)
{
  (*dest) = exp(-(*src)) / pow(1 + exp(-(*src)), 2);
}

double square_loss(const mat *expected, const mat *layer)
{
  double loss = 0;
  size_t i;
  for (i=0; i < layer->size; i++)
    loss += pow(expected->data[i] - layer->data[i], 2);
  return loss;
}

void nabla_square_loss(mat *delta, const mat *layer, const mat *expected)
{
  mat_sub(delta, layer, expected);
  mat_scalar(delta, 2.0);
}


void nn_create(neural_network *nn, const size_t size, const size_t *shape)
{
  nn->size = size;
  nn->shape = (size_t *)malloc(sizeof(size_t) * size);
  memcpy(nn->shape, shape, sizeof(size_t) * size);
  nn->layers  = (mat *)malloc(sizeof(mat) * nn->size);
  nn->z       = (mat *)malloc(sizeof(mat) * (nn->size - 1));
  nn->deltas  = (mat *)malloc(sizeof(mat) * (nn->size - 1));
  nn->biases  = (mat *)malloc(sizeof(mat) * (nn->size - 1));
  nn->weights = (mat *)malloc(sizeof(mat) * (nn->size - 1));
  nn->nabla_w = (mat *)malloc(sizeof(mat) * (nn->size - 1));
  nn->nabla_b = (mat *)malloc(sizeof(mat) * (nn->size - 1));
  size_t i;
  for (i=0; i < nn->size; i++)
  {
    if (i > 0)
    {
      mat_create(&(nn->z[i-1]),       shape[i], 1);
      mat_create(&(nn->deltas[i-1]),  shape[i], 1);
      mat_create(&(nn->weights[i-1]), shape[i], shape[i-1]);
      mat_create(&(nn->biases[i-1]),  shape[i], 1);
      mat_create(&(nn->nabla_w[i-1]), shape[i], shape[i-1]);
      mat_create(&(nn->nabla_b[i-1]), shape[i], 1);
    }
    mat_create(&(nn->layers[i]), shape[i], 1);
  }
}

int nn_load(neural_network *nn, const char *file_name)
{
  FILE *file = fopen(file_name, "rb");
  if (file == NULL)
  {
    printf("ERROR: Could not load file.\n");
    return -1;
  }
  size_t size; 
  fread(&size, sizeof(size_t), 1, file);
  size_t shape[size];
  fread(shape, sizeof(size_t), size, file);
  nn_create(nn, size, shape);
  size_t i;
  
  // Get max weight size
  size_t max_size = 0;
  size_t msize;
  for (size_t i=0; i < nn->size-1; i++)
  {
    msize = shape[i] * shape[i+1];
    if (msize > max_size)
      max_size = msize;
  }
  
  double buffer[max_size];
  size_t wsize;
  size_t bsize;
  for (i=0; i < nn->size-1; i++)
  {
    wsize = shape[i] * shape[i+1];
    bsize = shape[i+1];
    fread(buffer, sizeof(double), wsize, file);
    mat_set_data(&nn->biases[i], buffer);
    fread(buffer, sizeof(double), bsize, file);
    mat_set_data(&nn->weights[i], buffer);
  }

  fclose(file);
  return 1;
}

void nn_free(neural_network *nn)
{
  nn->size = 0;
  
  size_t i;
  for (i=0; i < nn->size; i++)
  {
    if (i > 0)
    {
      mat_free(&(nn->z[i-1]));
      mat_free(&(nn->deltas[i-1]));
      mat_free(&(nn->weights[i-1]));
      mat_free(&(nn->biases[i-1]));
      mat_free(&(nn->nabla_w[i-1]));
      mat_free(&(nn->nabla_b[i-1]));
    }
    mat_free(&(nn->layers[i]));
  }

  free(nn->shape);
  free(nn->layers);
  free(nn->biases);
  free(nn->weights);
  nn->shape = NULL;
  nn->layers = NULL;
  nn->biases = NULL;
  nn->weights = NULL;
}

void nn_randomize(neural_network *nn)
{
  size_t i;
  for (i=0; i < nn->size - 1; i++)
     mat_rand(&(nn->weights[i]));
}

double nn_loss(neural_network *nn, const mat *Y)
{
  return square_loss(&nn->layers[nn->size-1], Y);
}

void nn_feedforward(neural_network *nn, const mat *input)
{
  mat_copy(&nn->layers[0], input);

  size_t i=0;
  for (i=0; i < nn->size - 1; i++)
  {
    mat_mult(&(nn->z[i]), &(nn->weights[i]), &(nn->layers[i]));
    mat_add(&(nn->z[i]), &(nn->z[i]), &(nn->biases[i]));
    mat_map(&(nn->layers[i+1]), &(nn->z[i]), sigmoid);
  }
}

void nn_backpropagate(neural_network *nn, const mat *expected)
{
  size_t i;

  mat *output         = &(nn->layers[nn->size - 1]);
  mat *output_delta   = &(nn->deltas[nn->size - 2]);
  mat *output_z       = &(nn->z[nn->size - 2]);
  mat *output_nabla_w = &(nn->nabla_w[nn->size - 2]);
  mat *output_nabla_b = &(nn->nabla_b[nn->size - 2]);

  mat_map(output_z, output_z, dsigmoid);
  nabla_square_loss(output_delta, output, expected);
  mat_had(output_delta, output_delta, output_z);
  
  mat layer_transposed;
  mat_create_transposed(&layer_transposed, &(nn->layers[nn->size - 2]));
  mat_mult(output_nabla_w, output_delta, &layer_transposed);
  mat_free(&layer_transposed);

  mat_copy(output_nabla_b, output_delta);
  
  for (i = nn->size - 2; i-- > 0; )
  {
    mat *delta        = &nn->deltas[i];
    mat *next_delta   = &nn->deltas[i + 1];
    mat *z            = &nn->z[i];
    mat *next_weights = &nn->weights[i + 1];
    mat *nabla_w      = &nn->nabla_w[i];
    mat *nabla_b      = &nn->nabla_b[i];

    mat_map(z, z, dsigmoid);
    // TODO: calling malloc every step is not optimal
    mat_create_transposed(&layer_transposed, next_weights);
    mat_mult(delta, &layer_transposed, next_delta);
    mat_had(delta, delta, z);
    mat_free(&layer_transposed);
    
    mat_create_transposed(&layer_transposed, &nn->layers[i]);
    mat_mult(nabla_w, delta, &layer_transposed);
    mat_free(&layer_transposed);

    mat_copy(nabla_b, delta);
  }
}

void nn_update_batch(neural_network *nn, const mat *X, const mat *Y, const size_t batch_size, const double eta)
{
  // TODO: multithreading
  size_t i, j;
  mat *sum_nabla_w = (mat *)malloc(sizeof(mat) * (nn->size - 1));
  mat *sum_nabla_b = (mat *)malloc(sizeof(mat) * (nn->size - 1));

  for (i=1; i < nn->size; i++) {
    mat_create(&sum_nabla_w[i-1], nn->shape[i], nn->shape[i-1]);
    mat_create(&sum_nabla_b[i-1], nn->shape[i], 1);
  }

  for (i=0; i < batch_size; i++)
  {
    nn_feedforward(nn, &X[i]);
    nn_backpropagate(nn, &Y[i]);

    for (j=0; j < nn->size-1; j++)
    {
      mat_add(&sum_nabla_w[j], &sum_nabla_w[j], &nn->nabla_w[j]);
      mat_add(&sum_nabla_b[j], &sum_nabla_b[j], &nn->nabla_b[j]);
    }
  }

  for (i=0; i < nn->size-1; i++)
  {
    mat_scalar(&sum_nabla_w[i], eta/(double)batch_size);
    mat_scalar(&sum_nabla_b[i], eta/(double)batch_size);
    mat_sub(&nn->weights[i], &nn->weights[i], &sum_nabla_w[i]);
    mat_sub(&nn->biases[i],  &nn->biases[i],  &sum_nabla_b[i]);
  }

  for (i=0; i < nn->size-1; i++)
  {
    mat_free(&sum_nabla_w[i]);
    mat_free(&sum_nabla_b[i]);
  }
  
  free(sum_nabla_w);
  free(sum_nabla_b);
}

void nn_print(neural_network *nn)
{
  size_t i, j, index;
  size_t max = 0;
  printf("shape: [ ");
  for(i=0; i < nn->size; i++)
  {
    printf("%zu ", nn->shape[i]);
    if (nn->shape[i] > max)
      max = nn->shape[i];
  }
  printf("]\n");
   
  for (i=0; i < max; i++)
  {
    for (j=0; j < nn->size; j++)
    {
      if (nn->shape[j] < i + 1)
        printf("          ");
      else
        printf("%.3e ", nn->layers[j].data[i]);
    }
    printf("\n");
  }
}

void nn_save(neural_network *nn, const char *file_name)
{
  FILE *file = fopen(file_name, "wb");
  if (file == NULL)
  {
    printf("ERROR: Could not load file.\n");
    return;
  }
  fwrite(&nn->size, sizeof(size_t), 1, file);
  fwrite(nn->shape, sizeof(size_t), nn->size, file);
  
  size_t i;
  for (i=0; i < nn->size-1; i++)
  {
    fwrite(nn->weights[i].data, sizeof(double), nn->weights[i].size, file);
    fwrite(nn->biases[i].data, sizeof(double), nn->biases[i].size, file);
  }

  fclose(file);
}

