#!/bin/bash

## Load Modules
module load dealii
module load intel/18.0.2
module load tau

## Set TAU parameters


## Configure
rm -rf CMakeCache.txt
cmake .


## Compile
make -j8
