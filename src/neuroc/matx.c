#include "matx.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define COMPARE(A, B) (A->rows == B->rows && A->cols == B->cols)

void matx_print_error(matx_error error)
{
  switch (error)
  {
    case MAT_OK:
      printf("\033[32mOK!\033[m\n");
      break;
    case MAT_ERR_NULL:
      printf("\033[31mERROR: matx_err_null\033[m\n");
      break;
    case MAT_ERR_ALLOC:
      printf("\033[31mERROR: matx_err_alloc\033[m\n");
      break;
    case MAT_ERR_SHAPE:
      printf("\033[31mERROR: matx_err_shape\033[m\n");
      break;
    case MAT_ERR_BOUNDS:
      printf("\033[31mERROR: matx_err_bounds\033[m\n");
      break;
    case MAT_ERR_OVERFLOW:
      printf("\033[31mERROR: matx_err_overflow\033[m\n");
      break;
    case MAT_WKS_ERR_FULL:
      printf("\033[31mERROR: matx_err_full\033[m\n");
      break;
  }
}

matx_error matxwks_new(matxwks *wks, size_t size)
{
  if (wks == NULL)
    return MAT_ERR_NULL;
  if (size > SIZE_MAX)
    return MAT_ERR_OVERFLOW;
  wks->size = size;
  wks->used = 0;
  wks->data = (double *)malloc(sizeof(double) * size);
  if (wks->data == NULL)
    return MAT_ERR_ALLOC;
  return MAT_OK;
}

matx_error matxwks_init(matxwks *wks, size_t size, double *data)
{
  if (wks == NULL)
    return MAT_ERR_NULL;
  if (size > SIZE_MAX)
    return MAT_ERR_OVERFLOW;
  wks->size = size;
  wks->used = 0;
  wks->data = data;
  return MAT_OK;
}

matx_error matxwks_reset(matxwks *wks)
{
  wks->used = 0;
  return MAT_OK;
}

matx_error matxwks_reg(matxwks *wks, matx* dst, size_t rows, size_t cols)
{
  if (wks->used + rows * cols > wks->size)
    return MAT_WKS_ERR_FULL;
  matx_init(dst, rows, cols, wks->data+wks->used);
  wks->used += rows * cols;
  return MAT_OK;
}

matx_error matxwks_free(matxwks *wks)
{ 
  free(wks->data);
  return MAT_OK;
}

matx_error matx_create(matx *src, const size_t rows, const size_t cols)
{
  if (src == NULL)
    return MAT_ERR_NULL;
  if (rows == 0 & cols == 0)
    return MAT_ERR_SHAPE;
  if (rows > SIZE_MAX / cols)
    return MAT_ERR_OVERFLOW;
  src->rows = rows;
  src->cols = cols;
  src->size = rows * cols;
  src->data = (double *)calloc(src->size, sizeof(double));
  if (src->data == NULL)
    return MAT_ERR_ALLOC;
  return MAT_OK;
}

matx_error matx_create_copy(matx *dest, const matx *src)
{
  if (src == NULL || dest == NULL)
    return MAT_ERR_NULL;
  dest->rows = src->rows;
  dest->cols = src->cols;
  dest->size = src->rows * src->cols;
  dest->data = (double *)malloc(src->size * sizeof(double));
  if (src->data == NULL)
    return MAT_ERR_ALLOC;
  memcpy(dest->data, src->data, sizeof(double) * dest->size);
  return MAT_OK;
}

matx_error matx_create_transposed(matx *dest, const matx *src)
{
  if (src == NULL || dest == NULL)
    return MAT_ERR_NULL;
  dest->rows = src->cols;
  dest->cols = src->rows;
  dest->size = src->rows * src->cols;
  dest->data = (double *)malloc(src->size * sizeof(double));
  if (src->data == NULL)
    return MAT_ERR_ALLOC;
  size_t i, j;
  for (i=0; i < dest->rows; i++)
  for (j=0; j < dest->cols; j++)
    dest->data[i * dest->cols + j] = src->data[j * src->cols + i];
  return MAT_OK;
}

matx_error matx_create_from_buffer(matx *src, const size_t rows, const size_t cols, const double *data)
{
  if (src == NULL)
    return MAT_ERR_NULL;
  if (rows == 0 & cols == 0)
    return MAT_ERR_SHAPE;
  if (rows > SIZE_MAX / cols)
    return MAT_ERR_OVERFLOW;
  src->rows = rows;
  src->cols = cols;
  src->size = rows * cols;
  src->data = (double *)malloc(src->size * sizeof(double));
  if (src->data == NULL)
    return MAT_ERR_ALLOC;
  memcpy(src->data, data, sizeof(double) * src->size);
  return MAT_OK;
}

matx_error matx_init(matx *src, const size_t rows, const size_t cols, double *data)
{
  if (src == NULL)
    return MAT_ERR_NULL;
  if (rows == 0 & cols == 0)
    return MAT_ERR_SHAPE;
  if (rows > SIZE_MAX / cols)
    return MAT_ERR_OVERFLOW;
  src->rows = rows;
  src->cols = cols;
  src->size = rows * cols;
  src->data = data;
  return MAT_OK;
}

matx_error matx_free(matx *src)
{
  if (src == NULL)
    return MAT_ERR_NULL;
  free(src->data);
  src->rows = 0;
  src->cols = 0;
  src->size = 0;
  src->data = NULL;
  return MAT_OK;
}

matx_error matx_free_array(matx *array, size_t size)
{
  if (array == NULL)
    return MAT_ERR_NULL;
  size_t i;
  matx_error ret;
  for (i=0; i < size; i++)
    ret = matx_free(array+i);
    if (ret != MAT_OK)
      return ret;
  free(array);
  return MAT_OK;
}

matx_error matx_set_data(matx *dest, const double *data)
{
  memcpy(dest->data, data, sizeof(double) * dest->size);
  return MAT_OK;
}

matx_error matx_copy(matx *dest, const matx *src)
{
  if (!COMPARE(dest, src))
    return MAT_ERR_SHAPE;
  memcpy(dest->data, src->data, sizeof(double) * dest->size);
  return MAT_OK;
}

matx_error matx_rand(matx *src)
{
  size_t i;
  for (i=0; i < src->size; i++)
    src->data[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
  return MAT_OK;
}

matx_error matx_index(size_t *dest, const matx *src, const size_t row, const size_t column)
{
  if (row >= src->rows || column >= src->cols)
    return MAT_ERR_BOUNDS;
  (*dest) = row * src->cols + column;
  return MAT_OK;
}

matx_error matx_get(double *dest, const matx *src, const size_t row, const size_t column)
{
  size_t index;
  matx_error ret = matx_index(&index, src, row, column);
  if (ret != MAT_OK)
    return ret;
  (*dest) = src->data[index];
  return MAT_OK;
}

matx_error matx_set(matx *dest, const double src, const size_t row, const size_t column)
{
  size_t index;
  matx_error ret = matx_index(&index, dest, row, column);
  if (ret != MAT_OK)
    return ret;
  dest->data[index] = src;
  return MAT_OK;
}

matx_error matx_add(matx *dest, const matx *A, const matx *B)
{
  if (!COMPARE(A, B) || !COMPARE(A, dest))
    return MAT_ERR_SHAPE;
  size_t i;
  for (i=0; i < A->rows * A->cols; i++)
    dest->data[i] = A->data[i] + B->data[i];
  return MAT_OK;
}

matx_error matx_sub(matx *dest, const matx *A, const matx *B)
{
  if (!COMPARE(A, B) || !COMPARE(A, dest))
    return MAT_ERR_SHAPE;
  size_t i;
  for (i=0; i < A->rows * A->cols; i++)
    dest->data[i] = A->data[i] - B->data[i];
  return MAT_OK;
}

matx_error matx_mul(matx *dest, const matx *A, const matx *B)
{
  if (A->cols != B->rows || dest->rows != A->rows || dest->cols != B->cols)
    return MAT_ERR_SHAPE;
  size_t i, j, k;
  double value;
  for (i=0; i < A->rows; i++)
  for (j=0; j < B->cols; j++)
  {
    value = 0;
    for (k=0; k < A->cols; k++)
      value += A->data[i * A->cols + k] * B->data[k * B->cols + j];
    dest->data[i * dest->cols + j] = value;
  }
  return MAT_OK;
}

matx_error matx_had(matx *dest, const matx *A, const matx *B)
{
  if (!COMPARE(A, B) || !COMPARE(A, dest))
    return MAT_ERR_SHAPE;
  size_t i;
  for (i=0; i < A->size; i++)
    dest->data[i] = A->data[i] * B->data[i];
  return MAT_OK;
}

matx_error matx_scalar(matx *dest, const double scalar)
{
  size_t i;
  for (i=0; i < dest->size; i++)
    dest->data[i] *= scalar;
  return MAT_OK;
}

matx_error matx_map(matx *dest, matx *src, void (*oper)(double *dest, const double *src))
{
  if (!COMPARE(dest, src))
    return MAT_ERR_SHAPE;
  size_t i;
  for (i = 0; i < dest->size; i++)
    oper(&dest->data[i], &src->data[i]);
  return MAT_OK;
}

matx_error matx_transpose(matx *dest, matx *src)
{
  if (dest->rows != src->cols || dest->cols != src->rows)
    return MAT_ERR_SHAPE;
  size_t i, j;
  for (i=0; i < dest->rows; i++)
  for (j=0; j < dest->cols; j++)
    dest->data[i * dest->cols + j] = src->data[j * src->cols + i];
  return MAT_OK;
}

void matx_print(const matx *src)
{
  size_t i, j;
  double value;
  printf("%zux%zu\n", src->rows, src->cols);
  for (i=0; i < src->rows; i++)
  {
    if (i == 0)
      printf("[ ");
    else
      printf("  ");
    for (j=0; j < src->cols; j++)
    {
      value = src->data[i * src->cols + j];
      printf("%.3e ", value);
    }
    if (i == src->rows - 1)
      printf("]");
    printf("\n");
  }
}

