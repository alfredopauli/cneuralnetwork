# NeuroC
A simple deep-learning library written in C from scratch.

## neuroc_multperc (Multi-layered perceptron).

```c
#include <neuro.h>

int main(void)
{
    neuroc_multperc *nn = ncml_alloc();
    ncml_init_random(nn);
    ncml_config_learn(nn, 4, 128, 0.1);
    
    size_t train_size;
    matx* train_X, train_Y;
    // Gather train_X and train_Y by some mean...
    
    size_t epochs = 10;
    size_t e;
    for (e=0; e < epochs; e++)
        ncml_learn(nn, train_X, train_Y, train_size);

    ncml_free(nn);
    return 0;
}
```
