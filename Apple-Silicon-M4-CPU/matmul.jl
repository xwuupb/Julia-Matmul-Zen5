using Base.Threads
using InteractiveUtils
using LinearAlgebra
using Printf
using StaticArrays

#
# This Julia code is for Apple Silicon M4 CPU
#
# L1D cache: 128 KiB = 131072 bytes
# here: 720 * 16 * Float64 = 92160 bytes
const ROW_CACHE = 720 # more is also possible to fit L1D
const K_CACHE   =  16
const COL_CACHE = 180
#
# SIMD vector registers (NEON): 32 V0 - V31, each 128-bit wide
# here (max): 5 * (4 + 1) + 4 = 29
const ROW_SIMD  =  10 # 5 * 2 Float64 (one 128-bit register)
const COL_SIMD  =   4

"""
    mymm!(c, a, b, n; j_s = 1, j_e = n)

Compute C = A * B (matrix multiplication) using cache and SIMD vector register blocking.

- `c`: Matrix C (column-major)
- `a`: Matrix A (column-major)
- `b`: Matrix B (column-major)
- `n`: Square nxn matrices
- `j_s`: First column for C (or B), default 1 for serial execution
- `j_e`: Last  column for C (or B), default n for serial execution
"""
function mymm!(c, a, b, n; j_s = 1, j_e = n)
  #
  # SIMD vector registers to store submatrix C
  #
  c_1 = MVector{ROW_SIMD, Float64}(undef)
  c_2 = MVector{ROW_SIMD, Float64}(undef)
  c_3 = MVector{ROW_SIMD, Float64}(undef)
  c_4 = MVector{ROW_SIMD, Float64}(undef)
  #
  # perform matrix multiplication
  #
  for j in j_s:COL_CACHE:j_e
    for k in 1:K_CACHE:n
      for i in 0:ROW_CACHE:n-1 # 0-based indexing
        #
        # cache blocking
        # submatrix A in L1D cache
        #
        for j_b in j:COL_SIMD:j+COL_CACHE-1
            for i_b in i:ROW_SIMD:i+ROW_CACHE-1
            #
            # SIMD vector register blocking
            #
            # 1. read submatrix C into SIMD registers
            @simd for i_r in 1:ROW_SIMD
              c_1[i_r] = c[i_b + i_r, j_b ]
            end
            j_b1 = j_b + 1
            @simd for i_r in 1:ROW_SIMD
              c_2[i_r] = c[i_b + i_r, j_b1]
            end
            j_b2 = j_b + 2
            @simd for i_r in 1:ROW_SIMD
              c_3[i_r] = c[i_b + i_r, j_b2]
            end
            j_b3 = j_b + 3
            @simd for i_r in 1:ROW_SIMD
              c_4[i_r] = c[i_b + i_r, j_b3]
            end
            #
            # 2. multiplication of submatrices in SIMD registers
            for k_r in k:k+K_CACHE-1
              @simd for i_r in 1:ROW_SIMD
                c_1[i_r] = muladd(a[i_b + i_r, k_r], b[k_r, j_b ], c_1[i_r])
                c_2[i_r] = muladd(a[i_b + i_r, k_r], b[k_r, j_b1], c_2[i_r])
                c_3[i_r] = muladd(a[i_b + i_r, k_r], b[k_r, j_b2], c_3[i_r])
                c_4[i_r] = muladd(a[i_b + i_r, k_r], b[k_r, j_b3], c_4[i_r])
              end
            end
            #
            # 3. store submatrix C to main memory
            # (here performance issue due to cache coherence problems in multithreading)
            @simd for i_r in 1:ROW_SIMD
              c[i_b + i_r, j_b ] = c_1[i_r]
            end
            @simd for i_r in 1:ROW_SIMD
              c[i_b + i_r, j_b1] = c_2[i_r]
            end
            @simd for i_r in 1:ROW_SIMD
              c[i_b + i_r, j_b2] = c_3[i_r]
            end
            @simd for i_r in 1:ROW_SIMD
              c[i_b + i_r, j_b3] = c_4[i_r]
            end
            #
            # end of SIMD blocking
            #
          end
        end
        #
        # end of cache blocking
        #
      end
    end
  end
end

"""
  mymm_mt!(c, a, b, n, nths)

Compute C = A * B (matrix multiplication) with multithreading.

- `c`: Matrix C (column-major)
- `a`: Matrix A (column-major)
- `b`: Matrix B (column-major)
- `n`: Square nxn matrices
- `nths`: Number of threads
"""
function mymm_mt!(c, a, b, n, nths)
  @threads :static for tid in 0:nths-1
    m = n ÷ nths # each thread computes m columns in matrix C
    mymm!(c, a, b, n; j_s = (tid * m) + 1, j_e = (tid + 1) * m)
  end
end

function go_main(n, freq, superscalar, simd, fma)
  InteractiveUtils.versioninfo()
  a = zeros(n, n)
  b = zeros(n, n)
  c = zeros(n, n)
  d = zeros(n, n)
  for j in 1:n
    for i in 1:n
      # for exact floating-point number operations
      a[i, j] = rand(UInt) % 32 / 32.0
      b[i, j] = rand(UInt) % 32 / 32.0
    end
  end
  @printf("\n")
  @printf("each %dx%d matrix: %6.2f GB\n", n, n, 1.0e-9 * sizeof(Float64) * n * n)
  @printf("CPU max clock: %6.2f GHz\n", freq)
  @printf("Superscalar: %4d\n", superscalar)
  @printf("SIMD vector length: %4d-bit\n", simd * sizeof(Float64) * 8)
  @printf("fused multiply-add: %4d\n", fma)
  #
  # single CPU core
  #
  nths = 1
  single_core_peak = freq * superscalar * simd * fma
  @printf("\nsingle CPU core HW peak performance (Float64): %6.2f GFLOPS", single_core_peak)
  @printf("\nsingle CPU core: Benchmarking ...\n\n")
  # my implementation
  @printf("Me: # of threads: %d\n", nths)
  mymm!(c, a, b, n) # warm-up
  fill!(c, 0.0)
  start = time_ns()
  mymm!(c, a, b, n)
  stop = time_ns()
  mymm_time = 1.0e-9 * (stop - start)
  mymm_perf = 2.0e-9 * n * n * n / mymm_time
  # DGEMM in BLAS
  BLAS.set_num_threads(nths)
  @printf("BLAS Info:\n")
  @printf("  Lib → %s\n", BLAS.get_config())
  @printf("  # of BLAS threads → %d\n", BLAS.get_num_threads())
  d = a * b # warm-up
  fill!(d, 0.0)
  start = time_ns()
  d = a * b
  stop = time_ns()
  blas_time = 1.0e-9 * (stop - start)
  blas_perf = 2.0e-9 * n * n * n / blas_time
  # report
  @printf("Me:       walltime: %8.3f sec -> %9.1f GFLOPS (%8.2f%% HW-peak, %8.2f%% OpenBLAS)\n",
           mymm_time, mymm_perf, 100.0 * mymm_perf / single_core_peak,
           100.0 * mymm_perf / blas_perf)
  @printf("OpenBLAS: walltime: %8.3f sec -> %9.1f GFLOPS (%8.2f%% HW-peak)\n",
           blas_time, blas_perf, 100.0 * blas_perf / single_core_peak)
  # verify correctness
  max_abs_error = -1.0
  for j in 1:n
    for i in 1:n
      t = abs(c[i, j] - d[i, j])
      max_abs_error = t > max_abs_error ? t : max_abs_error
    end
  end
  @printf("max abs error: %15.8e\n", max_abs_error)
  #
  # multi CPU cores
  #
  nths = nthreads(:default)
  multi_cores_peak = single_core_peak * nths
  @printf("\nmulti CPU cores HW peak performance (Float64): %6.2f GFLOPS", multi_cores_peak)
  @printf("\nmulti CPU cores: Benchmarking ...\n\n")
  # my implementation
  @printf("Me: # of threads: %d\n", nths)
  mymm_mt!(c, a, b, n, nths) # warm-up
  fill!(c, 0.0)
  start = time_ns()
  mymm_mt!(c, a, b, n, nths)
  stop = time_ns()
  mymm_mt_time = 1.0e-9 * (stop - start)
  mymm_mt_perf = 2.0e-9 * n * n * n / mymm_mt_time
  # DGEMM in BLAS
  BLAS.set_num_threads(nths)
  @printf("BLAS Info:\n")
  @printf("  Lib → %s\n", BLAS.get_config())
  @printf("  # of BLAS threads → %d\n", BLAS.get_num_threads())
  d = a * b # warm-up
  fill!(d, 0.0)
  start = time_ns()
  d = a * b
  stop = time_ns()
  blas_mt_time = 1.0e-9 * (stop - start)
  blas_mt_perf = 2.0e-9 * n * n * n / blas_mt_time
  # report
  @printf("Me:       walltime: %8.3f sec -> %9.1f GFLOPS (%8.2f%% HW-peak, %8.2f%% OpenBLAS)\n",
           mymm_mt_time, mymm_mt_perf, 100.0 * mymm_mt_perf / multi_cores_peak,
           100.0 * mymm_mt_perf / blas_mt_perf)
  @printf("OpenBLAS: walltime: %8.3f sec -> %9.1f GFLOPS (%8.2f%% HW-peak)\n",
           blas_mt_time, blas_mt_perf, 100.0 * blas_mt_perf / multi_cores_peak)
  # verify correctness
  max_abs_error = -1.0
  for j in 1:n
    for i in 1:n
      t = abs(c[i, j] - d[i, j])
      max_abs_error = t > max_abs_error ? t : max_abs_error
    end
  end
  @printf("max abs error: %15.8e\n", max_abs_error)
end

function main_driver()
  for i in 1:24
    go_main(720*i, 4.4, 4, 2, 2)
    @printf("\nLet CPUs take a break ...\n\n")
    flush(stdout)
    sleep(16)
  end
end

main_driver()
