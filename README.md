# guff-matrix

This crate uses SIMD code to achieve very fast Galois Field matrix
multiplication.




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

- [ ] Arm NEON implementation of parallel long (bitwise) multiplication

- [ ] 4-way armv6 (Thumb) implementation of the long multiplication routine

Support for Arm targets requires nightly Rust build.

## Infinite Tape (Simulation)

Before I start writing arch-specific implementations, I'm focusing on
clearly documenting how the algorithm works. I'm going to implement a
non-SIMD version that uses the same basic ideas, but using a more
rusty style (infinite iterators). That's in `src/arch.rs` and can be
enabled as a feature:

    cargo test --features simulator --tests simulator

I'll also use this to prove that the algorithm works as intended.

- [x] Write and test simulation of non SIMD algorithm

- [x] Write and test simulation of SIMD algorithm


## Matrix multiplication

Using the simd version of the field multiplication routine, I now
have:

- [x] SIMD version of x86 matrix multiply

It needs a bit more work, but it's tested and runs around 3x faster
than the reference version. See `benches/vector_mul.rs` for
details. To run that with all relevant optimisations, you might need
to turn on some compile flags:

    RUSTFLAGS="-O -C target-cpu=native -C target-feature=+ssse3,+sse4.1,+sse4.2,+avx" cargo bench -q "matrix" 

