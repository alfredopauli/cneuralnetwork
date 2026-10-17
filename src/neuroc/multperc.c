#include "neuroc/multperc.h"
#include "neuroc/matx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <math.h>


static inline size_t min(size_t a, size_t b)
{
  return (a < b) ? (a) : (b);
}

struct neuroc_multperc {
  size_t n_layers;
  size_t *shape;
  matx *weights;
  matx *biases;
  matx pred;
};


//// Activation Functions:
  // Sigmoid activation function.
  void sigmoid(double *dest, const double *src)
  {
    (*dest) = 1.0 / (1.0 + exp(-*src));
  }
  
  // Sigmoid activation derivative.
  void dsigmoid(double *dest, const double *src)
  {
    double s = 1.0 / (1.0 + exp(-*src));
    (*dest) = s * (1.0 - s);
  }

//// Loss Functions:
  // Square loss function.
  double square_loss(const matx *expected, const matx *layer)
  {
    double loss = 0;
    size_t i;
    for (i=0; i < layer->size; i++)
      loss += pow(expected->data[i] - layer->data[i], 2);
    return loss;
  }
  
  // Square loss derivative.
  void nabla_square_loss(matx *delta, const matx *layer, const matx *expected)
  {
    matx_sub(delta, layer, expected);
    matx_scalar(delta, 2.0);
  }


//// INITIALIZATION:
  // Allocate memory for a new neural network object.
  neuroc_multperc *ncmp_alloc(void)
  {
    return (neuroc_multperc*)malloc(sizeof(neuroc_multperc));
  }
  // Free neural network memory.
  void ncmp_free(neuroc_multperc *nn)
  {
    free(nn->shape);
    matx_free_array(nn->weights,nn->n_layers-1);
    matx_free_array(nn->biases, nn->n_layers-1);
    matx_free(&nn->pred);
  }
  // Create new zero-initialized neural network with a given shape.
  void ncmp_init(
    neuroc_multperc *nn, const size_t n_layers, const size_t *shape
  )
  {
    nn->n_layers = n_layers;
    nn->shape    = (size_t *)malloc(sizeof(size_t) * n_layers);
    nn->biases   = (matx *)malloc(sizeof(matx) * (nn->n_layers - 1));
    nn->weights  = (matx *)malloc(sizeof(matx) * (nn->n_layers - 1));
    memcpy(nn->shape, shape, sizeof(size_t) * n_layers);
    size_t i, size;
    for (i=0; i < nn->n_layers-1; i++)
    {
      matx_create(&nn->weights[i], shape[i+1], shape[i]);
      matx_create(&nn->biases[i],  shape[i+1], 1);
    }
    matx_create(&nn->pred, shape[nn->n_layers - 1], 1);
  }
  // Randomize weights and biases of the neural network.
  void ncmp_init_random(
    neuroc_multperc *nn, const size_t n_layers, const size_t *shape
  )
  {
    printf("\033[34m[NN]\033[mInitializing random weights.\n");
    ncmp_init(nn, n_layers, shape);
    size_t i;
    for (i=0; i < nn->n_layers - 1; i++)
       matx_rand(&nn->weights[i]);
  }

//// DISK MANAGEMENT
  // TODO: Handle errors.
  // Load weights and biases from a file and create a neural network with these values.
  int ncmp_load(neuroc_multperc *nn, const char *file_name)
  {
    FILE *file = fopen(file_name, "rb");
    if (file == NULL)
    {
      printf("ERROR: Could not load file.\n");
      return -1;
    }
    size_t n_layers;
    fread(&n_layers, sizeof(size_t), 1, file);
    size_t shape[n_layers];
    fread(shape, sizeof(size_t), n_layers, file);
    ncmp_init(nn, n_layers, shape);
    size_t i;
    // Get max weight size
    size_t max_size = 0;
    size_t msize;
    for (size_t i=0; i < nn->n_layers-1; i++)
    {
      msize = shape[i] * shape[i+1];
      if (msize > max_size)
        max_size = msize;
    }
    double buffer[max_size];
    size_t wsize;
    size_t bsize;
    for (i=0; i < nn->n_layers-1; i++)
    {
      wsize = shape[i] * shape[i+1];
      bsize = shape[i+1];
      fread(buffer, sizeof(double), wsize, file);
      matx_set_data(&nn->biases[i], buffer);
      fread(buffer, sizeof(double), bsize, file);
      matx_set_data(&nn->weights[i], buffer);
    }
    fclose(file);
    return 1;
  }
  // Save neural network to a file.
  void ncmp_save(neuroc_multperc *nn, const char *file_name)
  {
    FILE *file = fopen(file_name, "wb");
    if (file == NULL)
    {
      printf("ERROR: Could not load file.\n");
      return;
    }
    fwrite(&nn->n_layers, sizeof(size_t), 1, file);
    fwrite(nn->shape, sizeof(size_t), nn->n_layers, file);
    size_t i;
    for (i=0; i < nn->n_layers-1; i++)
    {
      fwrite(nn->weights[i].data, sizeof(double), nn->weights[i].size, file);
      fwrite(nn->biases[i].data, sizeof(double), nn->biases[i].size, file);
    }
    fclose(file);
  }

//// Neural Network Algorithms:
  // Propagate the input forward.
  void ncmp_feedforward(
    neuroc_multperc *nn, const matx *input, matx *z, matx *a
  )
  {
    matx_copy(&a[0], input);
    size_t i=0;
    for (i=0; i < nn->n_layers - 1; i++)
    {
      matx_mul(&z[i],   &nn->weights[i], &a[i]);
      matx_add(&z[i],   &z[i],           &nn->biases[i]);
      matx_map(&a[i+1], &z[i],           sigmoid);
    }
  }
  // Propagate the error backwards.
  void ncmp_backpropagate(
    neuroc_multperc *nn, const matx *expected,
    matx *nabla_w, matx *nabla_b,
    matxwks *wks,
    matx *z, matx *a
  )
  {
    size_t i;
    
    matx *delta = (matx *)malloc(sizeof(matx) * (nn->n_layers - 1));
    for (i=0; i < nn->n_layers - 1; i++)
      matx_create(&delta[i], nn->shape[i+1], 1);

    matx zsigmoid;
    matx next_delta;
    matx trans;
    matx matxmul;
   
    matx *prev_a         = &a[nn->n_layers - 2];
    matx *output_a       = &a[nn->n_layers - 1];
    matx *output_z       = &z[nn->n_layers - 2];
    matx *output_delta   = &delta[nn->n_layers - 2];
    matx *output_nabla_w = &nabla_w[nn->n_layers - 2];
    matx *output_nabla_b = &nabla_b[nn->n_layers - 2];
  
    // σ'(z^L)
    matxwks_reg_s(wks, &zsigmoid, nn->shape[nn->n_layers - 2], 1);
    matx_map(&zsigmoid, output_z, dsigmoid);
    // ẟ = ∇C
    nabla_square_loss(output_delta, output_a, expected);
    // ẟ = ẟ ⊙ σ'(z^L) = ∇C ⊙ σ'(z^L)
    matx_had(output_delta, output_delta, &zsigmoid);
   
    // a_(L-1)^T
    matxwks_reg_s(wks, &trans, 1, nn->shape[nn->n_layers - 2]);
    matx_transpose(&trans, prev_a);
    // ∇w = ẟ * a_(L-1)^T
    matx_mul(output_nabla_w, output_delta, &trans);

    // ∇b = ẟ
    matx_copy(output_nabla_b, output_delta);

    for (i = nn->n_layers - 2; i-- > 0; )
    {
      matxwks_reset(wks);

      matx *a_prev       = &a[i];
      matx *z_curr       = &z[i];
      matx *delta_curr   = &delta[i];
      matx *delta_next   = &delta[i+1];
      matx *nabla_w_curr = &nabla_w[i];
      matx *nabla_b_curr = &nabla_b[i];
      matx *w_next       = &nn->weights[i+1];
      
      // σ'(z^L)
      matxwks_reg_s(wks, &zsigmoid, nn->shape[i+1], 1);
      matx_map(&zsigmoid, z_curr, dsigmoid);
      // w^(L+1)^T 
      matxwks_reg_s(wks, &trans, nn->shape[i+1], nn->shape[i+2]);
      matx_transpose(&trans, w_next);
      // w^(L+1)^T ẟ^(L+1)
      matxwks_reg_s(wks, &matxmul, nn->shape[i+1], 1);
      matx_mul(&matxmul, &trans, delta_next);
      // ẟ^L = (w^(L+1)^T ẟ^(L+1)) ⊙ σ'(z^L)
      matx_had(delta_curr, &matxmul, &zsigmoid);
      
      matxwks_reset(wks);

      // a_(L-1)^T
      matxwks_reg_s(wks, &trans, 1, nn->shape[i]);
      matx_transpose(&trans, prev_a);
      // ∇w = ẟ * a_(L-1)^T
      matx_mul(nabla_w_curr, delta_curr, &trans);

      // ∇b = ẟ
      matx_copy(nabla_b_curr, delta_curr);
    }

    matx_free_array(delta, nn->n_layers-1);
  }

//// Learning:
  // Evaluate nabla_w and nabla_b.
  void ncml_evaluate_nabla(
    neuroc_multperc *nn, 
    const matx *X, const matx *Y, 
    const size_t start, const size_t end, 
    matx *sum_nabla_w, matx *sum_nabla_b,
    matx *nabla_w, matx *nabla_b,
    matxwks *wks, matx *z, matx *a
  )
  {
    size_t i, j;
    for (i=start; i <= end; i++)
    {
      ncmp_feedforward(nn, &X[i], z, a);
      ncmp_backpropagate(nn, &Y[i], &nabla_w[i], &nabla_b[i], wks, z, a);
      for (j=0; j < nn->n_layers-1; j++)
      {
        matx_add(&sum_nabla_w[j], &sum_nabla_w[j], &nabla_w[start + j]);
        matx_add(&sum_nabla_b[j], &sum_nabla_b[j], &nabla_b[start + j]);
      }
    }
  }
  // Update weights and biases for one batch.
  void ncml_update_batch(
    neuroc_multperc *nn, 
    const matx *X, const matx *Y, 
    const size_t batch_size, 
    const double eta, 
    const size_t threads,
    matxwks *wks, matx *z, matx *a
  )
  {
    pthread_t proc[threads];
    size_t t;
    for (t=0; t < threads; t++)
      pthread_create(&proc[t], NULL, nn_evaluate_nabla, evaluate_nabla_args);

    for (t=0; t < threads; t++)
      pthread_join(proc[i], NULL);
  
    size_t i;
    for (i=0; i < nn->n_layers-1; i++)
    {
      matx_scalar(&sum_nabla_w[i], eta/(double)batch_size);
      matx_scalar(&sum_nabla_b[i], eta/(double)batch_size);

      matx_sub(&nn->weights[i], &nn->weights[i], &sum_nabla_w[i]);
      matx_sub(&nn->biases[i],  &nn->biases[i],  &sum_nabla_b[i]);
    }
  
    for (i=0; i < nn->n_layers-1; i++)
    {
      matx_free(&sum_nabla_w[i]);
      matx_free(&sum_nabla_b[i]);
      matx_free(&nabla_w[i]);
      matx_free(&nabla_b[i]);
    }

    free(sum_nabla_w);
    free(sum_nabla_b);
    free(nabla_w);
    free(nabla_b);
  }
  // Update weights and biases.
  void ncml_learn(
    neuroc_multperc *nn, 
    const matx *X, const matx *Y, 
    const size_t train_size,
    const size_t batch_size, 
    const double eta, 
    const size_t threads
  )
  {
    size_t i;
    size_t t;
    size_t index;

    size_t nabla_size = nn->n_layers - 1;
    size_t z_size = nn->n_layers - 1;
    size_t a_size = nn->n_layers;
   
    printf("\033[34m[NN]\033[m LOG: Creating shared workspace.\n");
    matxwks wks[threads];
    matx *z           = (matx *)malloc(sizeof(matx) * z_size     * threads);
    matx *a           = (matx *)malloc(sizeof(matx) * a_size     * threads);
    matx *nabla_w     = (matx *)malloc(sizeof(matx) * nabla_size * threads);
    matx *nabla_b     = (matx *)malloc(sizeof(matx) * nabla_size * threads);
    matx *nabla_w_sum = (matx *)malloc(sizeof(matx) * nabla_size);
    matx *nabla_b_sum = (matx *)malloc(sizeof(matx) * nabla_size);
    
    for (t=0; t < threads; t++)
    {  
      matxwks_init(&wks[t], nn->max_size * 3);
      for  (i=0; i < z_size; i++)
      {
        index = t*z_size + i;
        matx_create(&z[index], nn->shape[i+1], 1);
      }
      for (i=0; i < a_size; i++)
      {
        index = t*a_size + i;
        matx_create(&a[index], nn->shape[i], 1);
      }
      for (i=0; i < nabla_size; i++) 
      {
        index = t*nabla_size + i;
        matx_create(&nabla_w_sum[index], nn->shape[i+1], nn->shape[i]);
        matx_create(&nabla_w[index],     nn->shape[i+1], nn->shape[i]);
        matx_create(&nabla_b_sum[index], nn->shape[i+1], 1);
        matx_create(&nabla_b[index],     nn->shape[i+1], 1);
      }
    }

    printf("\033[34m[NN]\033[m LOG: training for %zu datapoints.\n", train_size);
    i = 0;
    size_t n;
    while (i < train_size)
    {
      n = min(batch_size, train_size - i);
      nn_update_batch(nn, X+i, Y+i, n, eta, threads, wks, z, a);
      i += n;
      printf("\r\033[2K\033[34m[NN]\033[m LOG: training %zu/%zu", i, train_size);
      fflush(stdout);
    }

    matx_free_array(a,           a_size     * threads);
    matx_free_array(z,           z_size     * threads);
    matx_free_array(nabla_w,     nabla_size * threads);
    matx_free_array(nabla_b,     nabla_size * threads);
    matx_free_array(nabla_w_sum, nabla_size);
    matx_free_array(nabla_b_sum, nabla_size);
    printf("\n");
  }

//// Evaluation:
  // Get prediction from neural network.
  const matx* ncml_predict(neuroc_multperc *nn, const matx *X)
  {
    size_t i;
    matx *z = (matx *)malloc(sizeof(matx) * (nn->n_layers-1));
    matx *a = (matx *)malloc(sizeof(matx) * nn->n_layers);
    for (i=0; i < nn->n_layers; i++)
    {
      if (i > 0)
        matx_create(&z[i-1], nn->shape[i], 1);
      matx_create(&a[i], nn->shape[i], 1);
    }
    ncml_feedforward(nn, X, z, a);
    matx_copy(&nn->pred, &a[nn->n_layers - 1]);
    matx_free_array(a, nn->n_layers);
    matx_free_array(z, nn->n_layers - 1);
    return &nn->pred;
  }

