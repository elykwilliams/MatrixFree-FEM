#!/bin/bash

module load dealii


#export NUM_THREADS=10
export REFINEMENT_LEVELS=2
export FE_DEGREE=3
export DIMENSION=3
export DO_CRS=1

#export OMP_PLACES=cores
#export OMP_PROC_BIND=close

# export GMON_OUT_PREFIX=profile_data

for R in 1 2 3 4
do
	for N in 24 48 96 144 288
	do
		export OMP_NUM_THREADS=${N}
		export RUNTIME_FLAGS="-C ${DO_CRS} -r ${R} -N ${N} -p ${FE_DEGREE} -d ${DIMENSION}"

		# perf stat -B -e cache-references,cache-misses,cycles,instructions ./MatrixFree ${RUNTIME_FLAGS} >> perf_${N}.txt
		
		./MatrixFree ${RUNTIME_FLAGS} >> Timing_C${DO_CRS}_R${R}.txt
	done
done
