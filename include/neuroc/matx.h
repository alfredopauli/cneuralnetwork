#ifndef __MAT_H
#define __MAT_H
#include <stdlib.h>
#include <assert.h>

#define matxwks_reg_s(wks, matx, rows, cols) assert(matxwks_reg((wks), (matx), (rows), (cols)) == MAT_OK)

// Matrix library

typedef enum {
  MAT_OK = 0,
  MAT_ERR_NULL,
  MAT_ERR_ALLOC,
  MAT_ERR_SHAPE,
  MAT_ERR_BOUNDS,
  MAT_ERR_OVERFLOW,
  MAT_WKS_ERR_FULL
} matx_error;

typedef struct {
  size_t rows;
  size_t cols;
  size_t size;
  double *data;
} matx;

typedef struct {
  size_t size;
  size_t used;
  double *data;
} matxwks;

void matx_print_error(matx_error error);

matx_error matxwks_new(matxwks *wks, size_t size);
matx_error matxwks_init(matxwks *wks, size_t size, double *data);
matx_error matxwks_reset(matxwks *wks);
matx_error matxwks_reg(matxwks *wks, matx* dst, size_t rows, size_t cols);
matx_error matxwks_free(matxwks *wks);

matx_error matx_create(matx *src, const size_t rows, const size_t cols);
matx_error matx_create_copy(matx *dest, const matx *src);
matx_error matx_create_transposed(matx *dest, const matx *src);
matx_error matx_create_from_buffer(matx *src, const size_t rows, const size_t cols, const double *data);
matx_error matx_init(matx *src, const size_t rows, const size_t cols, double *data);
matx_error matx_free(matx *src);
matx_error matx_free_array(matx *array, size_t size);
matx_error matx_set_data(matx *src, const double *data);
matx_error matx_copy(matx *dest, const matx *src);
matx_error matx_rand(matx *src);
matx_error matx_index(size_t *dest, const matx *src, const size_t row, const size_t column);
matx_error matx_get(double *dest, const matx *src, const size_t row, const size_t column);
matx_error matx_set(matx *dest, const double src, const size_t row, const size_t column);
matx_error matx_add(matx *dest, const matx *A, const matx *B);
matx_error matx_sub(matx *dest, const matx *A, const matx *B);
matx_error matx_mul(matx *dest, const matx *A, const matx *B);
matx_error matx_had(matx *dest, const matx *A, const matx *B);
matx_error matx_scalar(matx *dest, const double scalar);
matx_error matx_map(matx *dest, matx *src, void (*oper)(double *dest, const double *src));
matx_error matx_transpose(matx *dest, matx *src);
void      matx_print(const matx *src);

#endif
