#!/bin/bash
#SBATCH --nodes=12
module load OpenMPI
module load OpenSSL
cd /home/richard/project2
gcc generator.c -o generator -lssl -L /usr/local/ssl/lib -lssl -lcrypto -Wall -lm -std=c99
mpicc -fopenmp "$USER-knapsack.c" -o "$USER-knapsack"

./generator 10 5 | mpiexec -n 12 "$USER-knapsack"