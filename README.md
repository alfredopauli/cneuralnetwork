```c
#include <neuro.c>

int main(void)
{
    neuroc *nn = nn_alloc();
    nn_init_random(nn);
    nn_config_learn(nn, 4, 128, 0.1);
    
    size_t train_size;
    matx* train_X, train_Y;
    // Gather train_X and train_Y by some mean...
    
    size_t epochs = 10;
    size_t e;
    for (e=0; e < epochs; e++)
        nn_learn(nn, train_X, train_Y, train_size);

    nn_free(nn);
    return 0;
}
```
