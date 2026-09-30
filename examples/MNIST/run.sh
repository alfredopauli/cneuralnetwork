#!/bin/bash

gcc -o main *.c ../../src/*.c -I../../include -lm
./main

