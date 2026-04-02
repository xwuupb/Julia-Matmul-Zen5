#!/usr/bin/env bash
#
# show system info
#
uname -a
system_profiler SPHardwareDataType
#
# run benchmarking
#
set -x
npcore=4
julia --cpu-target=apple-m4 --optimize=3 --check-bounds=no \
      --math-mode=user --project=@. -t ${npcore},0 -- matmul.jl
