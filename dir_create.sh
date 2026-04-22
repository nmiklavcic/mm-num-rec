#!/bin/bash

mkdir -p svd_matrices

for i in $(seq 1 59); do
    mkdir -p "svd_matrices/k_$i"
done
