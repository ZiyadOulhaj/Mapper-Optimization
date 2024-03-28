#!/bin/bash

oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./Single_cell_sctda.sh sctda_5      Datasets/scTDA/     3 25 0.3   1e-5   5e-4 10 1   200 10  0 1 0"
oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./Single_cell_sctda.sh sctda_4      Datasets/scTDA/     3 25 0.3   1e-4   5e-4 10 1   200 10  0 1 0"
oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./Single_cell_sctda.sh sctda_3      Datasets/scTDA/     3 25 0.3   1e-3   5e-4 10 1   200 10  0 1 0"
oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./Single_cell_sctda.sh sctda_2      Datasets/scTDA/     3 25 0.3   1e-2   5e-4 10 1   200 10  0 1 0"
oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./Single_cell_sctda.sh sctda_1      Datasets/scTDA/     3 25 0.3   1e-1   5e-4 10 1   200 10  0 1 0"
#oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./Single_cell_sctda.sh sctda0       Datasets/scTDA/     3 25 0.3   1e0    5e-4 10 1   200 10  0 1 0"
#oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./Single_cell_sctda.sh sctda1       Datasets/scTDA/     3 25 0.3   1e1    5e-4 10 1   200 10  0 1 0"
#oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./Single_cell_sctda.sh sctda2       Datasets/scTDA/     3 25 0.3   1e2    5e-4 10 1   200 10  0 1 0"
