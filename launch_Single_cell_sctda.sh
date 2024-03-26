#!/bin/bash

oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./Single_cell_sctda.sh sctda      Datasets/scTDA/     3 25 0.3   1e-5   5e-4 10 1   200 10  0 1 0"
