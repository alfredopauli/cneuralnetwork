#!/bin/bash

set -e
gcc -o main train.c ../../src/*.c -I../../include -lm -lraylib
./main
