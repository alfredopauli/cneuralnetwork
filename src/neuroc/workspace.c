#include "neuroc/workspace.h"
#include <string.h>
#include <stdio.h>
#include <pthread.h>

size_t get_arg(workspace* wks, const char *k)
{
  size_t i;
  for (i=0; i < wks->n_elements; i++)
    if (strcmp(wks->keys[i], k) == 0)
      return i;
  return 0;
  printf("ÈRROR: could not find key");
}

void wks_init(
  workspace *wks, 
  const size_t n_elements, 
  const size_t *sizes, 
  const char **keys, 
  size_t copies
)
{
  wks->total_size     = 0;
  wks->total_elements = n_elements * copies;
  wks->n_elements     = n_elements;
  wks->keys           = (char **)malloc(sizeof(char *) * n_elements);
  wks->locks          = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * wks->n_elements);
  wks->offsets        = (size_t *)malloc(sizeof(size_t) * wks->n_elements);
  size_t i, index, len;
  for (i=0; i < n_elements * copies; i++)
  {
    if (i / n_elements == 0)
    {
      len = strlen(keys[i]);
      wks->keys[i] = (char *)malloc(len+1);
      strncpy(wks->keys[i], keys[i], len);
    }
    index = i % n_elements;
    wks->offsets[i] = wks->total_size;
    wks->total_size += sizes[index];
  }
  wks->data = (double *)malloc(sizeof(double) * wks->total_size);
}

double *wks_use(workspace *wks, const char *key, size_t n)
{
  size_t arg = get_arg(wks, key);
  pthread_mutex_lock(&wks->locks[n*wks->n_elements + arg]);
}

void wks_unuse(workspace *wks, const char *key, size_t n)
{
  size_t arg = get_arg(wks, key);
  pthread_mutex_unlock(&wks->locks[n*wks->n_elements + arg]);
}

void wks_free(workspace *wks)
{
  size_t i;
  for (i=0; i < wks->n_elements; i++)
    free(wks->keys[i]);

  free(wks->keys);
  free(wks->locks);
  free(wks->offsets);
  free(wks->data);
}
