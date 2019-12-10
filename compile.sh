#!/bin/bash

## Load Modules
module load dealii
module load intel/18.0.2

## Configure
rm -rf ./build
mkdir build
cd build 
cmake ..


## Compile
make -j48
