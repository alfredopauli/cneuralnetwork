#ifndef __MAT_H
#define __MAT_H
#include <stdlib.h>

typedef enum {
  MAT_OK = 0,
  MAT_ERR_NULL,
  MAT_ERR_MALLOC,
  MAT_ERR_SHAPE,
  MAT_ERR_BOUNDS,
  MAT_ERR_OVERFLOW
} mat_error;

typedef struct {
  size_t rows;
  size_t cols;
  size_t size;
  double *data;
} mat;

mat_error mat_create(mat *src, const size_t rows, const size_t cols);
mat_error mat_create_copy(mat *dest, const mat *src);
mat_error mat_create_transposed(mat *dest, const mat *src);
mat_error mat_create_from_buffer(mat *src, const size_t rows, const size_t cols, const double *data);
mat_error mat_free(mat *src);
mat_error mat_free_array(mat *array, size_t size);
mat_error mat_set_data(mat *src, const double *data);
mat_error mat_copy(mat *dest, const mat *src);
mat_error mat_rand(mat *src);
mat_error mat_index(size_t *dest, const mat *src, const size_t row, const size_t column);
mat_error mat_get(double *dest, const mat *src, const size_t row, const size_t column);
mat_error mat_set(mat *dest, const double src, const size_t row, const size_t column);
mat_error mat_add(mat *dest, const mat *A, const mat *B);
mat_error mat_sub(mat *dest, const mat *A, const mat *B);
mat_error mat_mult(mat *dest, const mat *A, const mat *B);
mat_error mat_had(mat *dest, const mat *A, const mat *B);
mat_error mat_scalar(mat *dest, const double scalar);
mat_error mat_map(mat *dest, mat *src, void (*oper)(double *dest, const double *src));
void      mat_print(const mat *src);

#endif
