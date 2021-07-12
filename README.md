# guff-matrix

This crate uses SIMD code to achieve very fast Galois Field matrix
multiplication.

The idea behind the matrix multiplication is to organise memory
accesses so that the SIMD engine can achieve 100% utilisation while
avoiding any non-aligned or partial vector reads. Output elements are
produced sequentially along the output matrix diagonal.

In order to achieve this, the input and output matrices are
constructed with a specific number of columns so that:

* the last vector read of the input matrix ends exactly at the matrix
  boundary

* when wrap-around at the output matrix occurs the algorithm starts
  filling a new diagonal

Essentially, the transform, input and output matrices are treated as
infinite tapes. We select input/output matrix width so that it has a
factor which is relatively prime to both of the transform matrix
dimensions k and n (to satisfy the second condition). We also
constrain the width of the input matrix to being a multiple of
`lcm(k,n,w,simd_width)` to satisfy the first condition. (`w` is the
word size, which for GF(2<sup>8</sup>) fields is `1`)

The matrices are organised in memory as follows:

* transform is row-major
* input is column-major
* output is either

In order to facilitate wrap-around at the end of the transform matrix,
if `k * n` is not a multiple of the SIMD width, we copy enough
elements from the start of the matrix to the end in order to make it a
multiple. This avoids the need to do conditional reads from memory.

The input matrix also has an extra simd_width bytes copied from the
start of the matrix. This is to deal with wrap-around for all
non-final reads. (?)

## Previous Implementation

I have previously implemented a version of this algorithm on a
PlayStation 3. It is available
[here](https://github.com/declanmalone/gnetraid/blob/master/PS3-IDA/08-fastmatrix/spu-matrix.c)



## SIMD Support

I will implement three different SIMD engines for field
multiplication across vectors:

- [x] x86 implementation of parallel long (bitwise) multiplication

- [x] Arm/Aarch64 NEON implementation using hardware polynomial
      multiply and table-based modular reduction (vmull/tvbl)

- [x] Arm NEON implementation of parallel long (bitwise) multiplication

I also have a 4-way armv6 (Thumb) implementation of the long
multiplication routine, which I may add for completeness. Its
performance is roughly comparable to doing four single multiply using
lookup tables, only slightly worse.

Support for Arm targets requires nightly Rust build.

## Infinite Tape (Simulation)

Before I start writing arch-specific implementations, I'm focusing on
clearly documenting how the algorithm works. I'm going to implement a
non-SIMD version that uses the same basic ideas, but using a more
rusty style (infinite iterators). That's in `src/arch.rs` and can be
enabled as a feature:

    cargo test --features fake-simd

I'll also use this to prove that the algorithm works as intended.

- [x] Write and test simulation of (non-vectorised) algorithm
