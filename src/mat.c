#include "mat.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define COMPARE(A, B) (A->rows == B->rows && A->cols == B->cols)


mat_error mat_create(mat *src, const size_t rows, const size_t cols)
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
    return MAT_ERR_MALLOC;
  return MAT_OK;
}

mat_error mat_create_copy(mat *dest, const mat *src)
{
  if (src == NULL || dest == NULL)
    return MAT_ERR_NULL;
  dest->rows = src->rows;
  dest->cols = src->cols;
  dest->size = src->rows * src->cols;
  dest->data = (double *)malloc(src->size * sizeof(double));
  if (src->data == NULL)
    return MAT_ERR_MALLOC;
  memcpy(dest->data, src->data, sizeof(double) * dest->size);
  return MAT_OK;
}

mat_error mat_create_transposed(mat *dest, const mat *src)
{
  if (src == NULL || dest == NULL)
    return MAT_ERR_NULL;
  dest->rows = src->cols;
  dest->cols = src->rows;
  dest->size = src->rows * src->cols;
  dest->data = (double *)malloc(src->size * sizeof(double));
  if (src->data == NULL)
    return MAT_ERR_MALLOC;
  size_t i, j;
  for (i=0; i < dest->rows; i++)
  for (j=0; j < dest->cols; j++)
    dest->data[i * dest->cols + j] = src->data[j * src->cols + i];
  return MAT_OK;
}

mat_error mat_create_from_buffer(mat *src, const size_t rows, const size_t cols, const double *data)
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
    return MAT_ERR_MALLOC;
  memcpy(src->data, data, sizeof(double) * src->size);
  return MAT_OK;
}

mat_error mat_free(mat *src)
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

mat_error mat_free_array(mat *array, size_t size)
{
  if (array == NULL)
    return MAT_ERR_NULL;
  size_t i;
  mat_error ret;
  for (i=0; i < size; i++)
    ret = mat_free(array+i);
    if (ret != MAT_OK)
      return ret;
  return MAT_OK;
}

mat_error mat_set_data(mat *dest, const double *data)
{
  memcpy(dest->data, data, sizeof(double) * dest->size);
  return MAT_OK;
}

mat_error mat_copy(mat *dest, const mat *src)
{
  if (!COMPARE(dest, src))
    return MAT_ERR_SHAPE;
  memcpy(dest->data, src->data, sizeof(double) * dest->size);
  return MAT_OK;
}

mat_error mat_rand(mat *src)
{
  size_t i;
  for (i=0; i < src->size; i++)
    src->data[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
  return MAT_OK;
}

mat_error mat_index(size_t *dest, const mat *src, const size_t row, const size_t column)
{
  if (row >= src->rows || column >= src->cols)
    return MAT_ERR_BOUNDS;
  (*dest) = row * src->cols + column;
  return MAT_OK;
}

mat_error mat_get(double *dest, const mat *src, const size_t row, const size_t column)
{
  size_t index;
  mat_error ret = mat_index(&index, src, row, column);
  if (ret != MAT_OK)
    return ret;
  (*dest) = src->data[index];
  return MAT_OK;
}

mat_error mat_set(mat *dest, const double src, const size_t row, const size_t column)
{
  size_t index;
  mat_error ret = mat_index(&index, dest, row, column);
  if (ret != MAT_OK)
    return ret;
  dest->data[index] = src;
  return MAT_OK;
}

mat_error mat_add(mat *dest, const mat *A, const mat *B)
{
  if (!COMPARE(A, B) || !COMPARE(A, dest))
    return MAT_ERR_SHAPE;
  size_t i;
  for (i=0; i < A->rows * A->cols; i++)
    dest->data[i] = A->data[i] + B->data[i];
  return MAT_OK;
}

mat_error mat_sub(mat *dest, const mat *A, const mat *B)
{
  if (!COMPARE(A, B) || !COMPARE(A, dest))
    return MAT_ERR_SHAPE;
  size_t i;
  for (i=0; i < A->rows * A->cols; i++)
    dest->data[i] = A->data[i] - B->data[i];
  return MAT_OK;
}

mat_error mat_mult(mat *dest, const mat *A, const mat *B)
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

mat_error mat_had(mat *dest, const mat *A, const mat *B)
{
  if (!COMPARE(A, B) || !COMPARE(A, dest))
    return MAT_ERR_SHAPE;
  size_t i;
  for (i=0; i < A->size; i++)
    dest->data[i] = A->data[i] * B->data[i];
  return MAT_OK;
}

mat_error mat_scalar(mat *dest, const double scalar)
{
  size_t i;
  for (i=0; i < dest->size; i++)
    dest->data[i] *= scalar;
  return MAT_OK;
}

mat_error mat_map(mat *dest, mat *src, void (*oper)(double *dest, const double *src))
{
  if (!COMPARE(dest, src))
    return MAT_ERR_SHAPE;
  size_t i;
  for (i = 0; i < dest->size; i++)
    oper(&dest->data[i], &src->data[i]);
  return MAT_OK;
}

void mat_print(const mat *src)
{
  size_t i, j;
  double value;
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

