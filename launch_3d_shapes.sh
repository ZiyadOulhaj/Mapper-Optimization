#!/bin/bash

oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./3d_shapes.sh human10       Human/10.off     3 25 0.3   0.01   0.05 10 0.01   200 10  0 1 0"
oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./3d_shapes.sh octopus132-3  Octopus/132.off  8 10 0.3   0.01   1.   10 0.05   200 10  0 1 0"
oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./3d_shapes.sh octopus132-4  Octopus/132.off  8 10 0.3   0.01   1.   10 0.05   200 10  0 1 0"
oarsub -t besteffort -t idempotent -l /core=10,walltime=24:00:00 "./3d_shapes.sh table142      Table/142.off    8 10 0.35  0.01   0.1  10 0.10   200 10  0 1 0"
