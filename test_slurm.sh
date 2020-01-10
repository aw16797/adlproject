#!/bin/bash

#SBATCH --job-name=soundClassification
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --mem=100M
