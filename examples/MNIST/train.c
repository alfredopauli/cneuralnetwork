#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "neuralnetwork.h"

static inline size_t min(size_t a, size_t b)
{
  return (a < b) ? (a) : (b);
}

// Function to swap bytes from Big-Endian to Little-Endian
uint32_t swap_uint32(uint32_t val) {
  return ((val >> 24) & 0xff) |      // Move byte 3 to byte 0
         ((val << 8)  & 0xff0000) |  // Move byte 1 to byte 2
         ((val >> 8)  & 0xff00) |    // Move byte 2 to byte 1
         ((val << 24) & 0xff000000); // Move byte 0 to byte 3
}

uint32_t read_normalized_mnist_image(mat **array, const char *file_name)
{
  printf("\033[1mLOG: Reading normalized values for MNIST input (images).\033[0m\n");

  FILE *file = fopen(file_name, "rb");
  if (file == NULL)
  {
    printf("ERROR: Could not open '%s'.\n", file_name);
    return 0;
  }

  uint32_t magic_number;
  uint32_t num_items;
  uint32_t rows;
  uint32_t cols;
  if ((fread(&magic_number, sizeof(uint32_t), 1, file) != 1) ||
      (fread(&num_items,    sizeof(uint32_t), 1, file) != 1) ||
      (fread(&rows,         sizeof(uint32_t), 1, file) != 1) ||
      (fread(&cols,         sizeof(uint32_t), 1, file) != 1))
  {
    printf("ERROR: Could not read header.\n");
    fclose(file);
    return 0;
  }

  magic_number = swap_uint32(magic_number);
  num_items    = swap_uint32(num_items);
  rows         = swap_uint32(rows);
  cols         = swap_uint32(cols);
  (*array) = (mat *)malloc(sizeof(mat) * num_items);

  printf("LOG: magic_number=%d\n", magic_number);
  printf("LOG: num_items   =%d\n", num_items);
  printf("LOG: rows        =%d\n", rows);
  printf("LOG: cols        =%d\n", cols);
  
  uint32_t i;
  uint32_t j;
  uint8_t data[rows * cols];
  for (i=0; i < num_items; i++)
  {
    printf("\r\033[2KLOG: processing element %d/%d", i + 1, num_items);
    if (fread(data, sizeof(uint8_t), rows * cols, file) != rows * cols)
    {
      printf("\nERROR: File reading terminated earlier than it was supposed to.\n");
      mat_free_array(*array, i);
      free(*array);
      return 0;
    }
    mat_create((*array)+i, rows * cols, 1);
    for (j=0; j < rows*cols; j++)
      (*array)[i].data[j] = (double)data[j] / 255.0;
  }
  printf("\n\033[32;1mLOG: Success!\033[m\n");
  
  fclose(file);

  return num_items;
}

uint32_t read_normalized_mnist_label(mat **array, const char *file_name)
{
  printf("\033[1mLOG: Reading normalized output values for MNIST output (labels).\033[0m\n");

  FILE *file = fopen(file_name, "rb");
  if (file == NULL)
  {
    printf("ERROR: Could not open '%s'.\n", file_name);
    return 0;
  }
  
  uint32_t magic_number;
  uint32_t num_items;
  if ((fread(&magic_number, sizeof(uint32_t), 1, file) != 1) ||
      (fread(&num_items,    sizeof(uint32_t), 1, file) != 1))
  {
    printf("ERROR: Could not read header.\n");
    fclose(file);
    return 0;
  }

  magic_number = swap_uint32(magic_number);
  num_items = swap_uint32(num_items);
  (*array) = (mat *)malloc(sizeof(mat) * num_items);

  printf("LOG: magic_number=%d\n", magic_number);
  printf("LOG: num_items   =%d\n", num_items);
  
  uint32_t i;
  uint8_t label;
  for (i=0; i < num_items; i++)
  {
    printf("\r\033[2KLOG: processing element %d/%d", i + 1, num_items);
    if (fread(&label, sizeof(uint8_t), 1, file) != 1)
    {
      printf("\nERROR: File reading terminated earlier than it was supposed to.\n");
      mat_free_array(*array, i);
      free(*array);
      return 0;
    }
    mat_create((*array)+i, 10, 1);
    (*array)[i].data[label] = 1;
  }
  printf("\n\033[32;1mLOG: Success!\033[m\n");

  fclose(file);

  return num_items;
}

uint8_t get_label_from_mat(mat *mat)
{
  size_t i;
  double prob;
  size_t max_label;
  double max_prob = -INFINITY;
  for (i=0; i < 10; i++)
  {
    prob = mat->data[i];
    if (prob > max_prob)
    {
      max_prob = prob;
      max_label = i;
    }
  }
  return max_label;
}

int main(void)
{
  srand(time(0));

  uint32_t train_size;
  uint32_t test_size;
  uint32_t size;

  mat *train_images;
  mat *train_labels;
  mat *test_images;
  mat *test_labels;
  
  train_size = read_normalized_mnist_image(
    &train_images, 
    "dataset/train-images.idx3-ubyte"
  );
  test_size = read_normalized_mnist_image(
    &test_images, 
    "dataset/t10k-images.idx3-ubyte"
  );

  size = read_normalized_mnist_label(
    &train_labels,
    "dataset/train-labels.idx1-ubyte"
  );
  if (size != train_size)
  {
    printf("ERROR: Number of labels does not match the number of images\n");
    mat_free_array(train_images, train_size);
    mat_free_array(test_images, test_size);
    mat_free_array(train_labels, size);
    return -1;
  }
  
  size = read_normalized_mnist_label(
    &test_labels,
    "dataset/t10k-labels.idx1-ubyte"
  );
  if (size != test_size)
  {
    printf("ERROR: Number of labels does not match the number of images\n");
    mat_free_array(train_images, train_size);
    mat_free_array(train_images, train_size);
    mat_free_array(train_labels, train_size);
    mat_free_array(test_images, test_size);
    mat_free_array(test_labels, size);
    return -1;
  }
  
  // Hyperparameters
  size_t epochs = 10;
  size_t batch_size = 256;
  double learning_step = 0.1;

  neural_network nn;
  const char *file_name = "model";
  const size_t shape[] = {784, 128, 64, 10};
  if (nn_load(&nn, file_name) == -1)
  {
    printf("LOG: Could not open Neural Network from file. Creating new randomized one.\n");
    nn_create(&nn, sizeof(shape) / sizeof(size_t), shape);
    nn_randomize(&nn);
  }
  
  size_t epoch;
  size_t i;
  size_t n;
  size_t correct_count;
  double prob;
  uint8_t j;
  double max_prob;
  uint8_t max_label;
  size_t index_max;
  float accuracy;
  for (epoch=0; epoch < epochs; epoch++)
  {
    printf("\033[1mLOG: epoch=%zu\033[0m\n", epoch + 1);
    // TODO: Shuffle training data before epoch
    i=0;
    while (i < train_size)
    {
      n = min(batch_size, train_size - i);
      nn_update_batch(&nn, train_images+i, train_labels+i, n, learning_step);
      i += n;
      printf("\r\033[2KLOG: training %zu/%d", i, train_size);
      fflush(stdout);
    }
    printf("\n");

    correct_count = 0;
    max_prob = -INFINITY;
    for (i=0; i < test_size; i++)
    {
      nn_feedforward(&nn, test_images+i);
      if (get_label_from_mat(&nn.layers[nn.size-1]) == get_label_from_mat(&test_labels[i]))
        correct_count++;
      printf("\r\033[2KLOG: testing %zu/%d", i + 1, test_size);
      fflush(stdout);
    }
    accuracy = 100.0*(float)correct_count/(float)test_size;
    printf("\n\033[33mLOG: accuracy \033[1m%.2f%%\n\033[m", accuracy);
    printf("LOG: Saving model. ");
    nn_save(&nn, file_name);
    printf("Success!\n");
  }  

  nn_free(&nn);

  mat_free_array(train_images, train_size);
  mat_free_array(train_labels, train_size);
  mat_free_array(test_images, test_size);
  mat_free_array(test_labels, test_size);

  return 0;
}

