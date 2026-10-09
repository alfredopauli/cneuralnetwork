#!/bin/bash

set -e
gcc -o main test.c ../../src/*.c -I../../include -lm -lraylib
./main

