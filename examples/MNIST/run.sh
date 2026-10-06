#!/bin/bash

gcc -o main test.c ../../src/*.c -I../../include -lm -lraylib
./main

