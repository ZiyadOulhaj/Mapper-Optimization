#!/bin/bash

oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh human10_run1       Human/10.off     3 25 0.3   0.01   0.05 10 0.01   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh human10_run2       Human/10.off     3 25 0.3   0.01   0.05 10 0.01   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh human10_run3       Human/10.off     3 25 0.3   0.01   0.05 10 0.01   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh human10_run4       Human/10.off     3 25 0.3   0.01   0.05 10 0.01   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh human10_run5       Human/10.off     3 25 0.3   0.01   0.05 10 0.01   200 10  0 1 0"

oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh octopus132_run1    Octopus/132.off  8 10 0.3   0.01   1.   10 0.05   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh octopus132_run2    Octopus/132.off  8 10 0.3   0.01   1.   10 0.05   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh octopus132_run3    Octopus/132.off  8 10 0.3   0.01   1.   10 0.05   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh octopus132_run4    Octopus/132.off  8 10 0.3   0.01   1.   10 0.05   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh octopus132_run5    Octopus/132.off  8 10 0.3   0.01   1.   10 0.05   200 10  0 1 0"

oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh table142_run1      Table/142.off    8 10 0.35  0.01   0.1  10 0.10   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh table142_run2      Table/142.off    8 10 0.35  0.01   0.1  10 0.10   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh table142_run3      Table/142.off    8 10 0.35  0.01   0.1  10 0.10   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh table142_run4      Table/142.off    8 10 0.35  0.01   0.1  10 0.10   200 10  0 1 0"
oarsub  -l /core=1,walltime=10:00:00 "./3d_shapes.sh table142_run5      Table/142.off    8 10 0.35  0.01   0.1  10 0.10   200 10  0 1 0"
