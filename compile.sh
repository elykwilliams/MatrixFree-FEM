#!/bin/bash

## Load Modules
module load dealii
module load intel/18.0.2

## Configure
rm -rf ./build
mkdir build
cd build 
cmake ..
cp ../run.sh .

## Compile
make -j48 release
