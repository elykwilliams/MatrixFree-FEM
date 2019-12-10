#!/bin/bash

module load dealii

export NUM_THREADS=4
export REFINEMENT_LEVELS=3
export FE_DEGREE=1
export DIMENSION=3
export DO_CRS=1

export OMP_PLACES=cores
export OMP_PROC_BIND=close

./MatrixFree -C ${DO_CRS} -r ${REFINEMENT_LEVELS} -N ${NUM_THREADS} -p ${FE_DEGREE} -d ${DIMENSION}

