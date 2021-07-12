//! Fast SIMD matrix multiplication for finite fields
//!
//! This crate implements two things:
//!
//! 1. Fast SIMD-based multiplication of vectors of finite field
//!    elements (GF(2<sup>8</sup>) with the polynomial 0x11b)
//!
//! 2. A (cache-friendly) matrix multiplication routine based on
//!    achieving 100% utilisation of the above
//!
//! This crate supports x86_64 and Arm (v7, v8) with NEON extensions.
//!
//! The matrix multiplication routine is heavily geared towards use in
//! implementing Reed-Solomon or Information Dispersal Algorithm
//! error-correcting codes.
//!
//! For x86_64 and Armv8 (Aarch64), building requires no extra
//! options:
//!
//! ```bash
//! cargo build
//! ```
//!
//! It seems that on armv7 platforms, the rust build system is unable
//! to detect the availability of `target_feature = ""`. As a
//! result, I've added "neon" as a build feature instead. Select it
//! with:
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --features neon
//! ```
//!
//! # Software Simulation Feature
//!
//! I've implemented a pure Rust version of the matrix multiplication
//! code. It uses the same basic idea as the optimised versions,
//! although for clarity, it works a byte at a time instead of
//! simulating SIMD multiplication on 8 or 16 bytes at a time.
//!
//! 
//!
//!
//!


#![feature(stdsimd)]


// Rationalise target arch/target feature/build feature
//
// I have three different arm-based sets of SIMD code:
//
// 1. thumb/dsp-based 4-way simd that works on armv6 and armv7, but
//    not, apparently, on armv8
//
// 2. neon-based 16-way reimplementation of the above, which works on
//    armv7 with neon extension, and armv8
//
// 3. new neon-based 8-way simd based on vmull and vtbl instructions,
//    which works on armv7 with neon extension, and armv8
//
// Since I'm controlling compilation by named features, I want all of
// these to be additive. As a result, I'll give each of them a
// separate module name, which will appear if the appropriate feature
// is enabled.
//

// Only one x86 implementation, included automatically
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

// Implementation (1) above
#[cfg_attr(all(target_arch = "arm"), feature = "arm_dsp")]
mod arm_dsp;

// Implementation (2) above
#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_long"))]
mod arm_long;

// Implementation (3) above
#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_vmull"))]
mod arm_vmull;

#[cfg(feature = "simulator")]
mod simulator;

pub fn gcd(mut a : usize, mut b : usize) -> usize {
  let mut t;
    loop {
	if b == 0 { return a }
	t = b;
	b = a % b;
	a = t;
    }
}

pub fn gcd3(a : usize, b : usize, c: usize) -> usize {
    gcd(a, gcd(b,c))
}

pub fn gcd4(a : usize, b : usize, c: usize, d : usize) -> usize {
    gcd(gcd(a,b), gcd(c,d))
}

pub fn lcm(a : usize, b : usize) -> usize {
    (a / gcd(a,b)) * b
}

pub fn lcm3(a : usize, b : usize, c: usize) -> usize {
    lcm( lcm(a,b), c)
}

pub fn lcm4(a : usize, b : usize, c: usize, d : usize) -> usize {
    lcm( lcm(a,b), lcm(c,d) )
}

// xform0, xform1 are externally stored registers; this tracks
// variables needed to read simd bytes at a time from memory into
// those registers, and to extract a full simd bytes for passing to
// multiply routine.
struct TransformTape {
    k : usize,
    n : usize,
    w : usize,
    xptr : usize,    // (next) read pointer within matrix
}

// My first macro. I think that it will be easier to write a generic
// version of the multiply routine that works across architectures if
// I can hide both register types (eg, _m128* on x86) and intrinsics.
//
// Another advantage is that I can test the macros separately.
//
// The only fly in the ointment is if I need fundamentally different
// logic to map operations onto intrinsics...

// #[macro_export]
macro_rules! new_xform_reader {
    ( $s:ident, $k:expr, $n:expr, $w:expr, $r0:ident, $r1:ident) => {
	let mut $s = TransformTape { k : $k, n : $n, w : $w, xptr : 0 };
    }
}

// Actually, I can eliminate explicit variable names above. That would
// solve the problem of having to use different number of variables to
// achieve a certain result.

// Only pass in that are germane to the algorithm, not the
// arch-specific implementation:
//
// init_xform_stream!(xform.as_ptr())
// init_input_stream!(input.as_ptr())
// init_output_stream!(output.as_ptr())
//
// ...

// Matrix sizes
//
// We multiply xform x input giving output
//
// These values are fixed by the transform matrix:
//
// * n: number of rows in xform
// * k: number of columns in xform
// * w: number of bytes in each element
//
// We also have simd_width, which is the width of the SIMD vectors, in
// bytes.
//
// The input matrix has k rows. The output matrix has n rows.
//
// We fix the input and output matrices as having the same number of
// columns, c. It has to have a factor f that is coprime to both kw
// and n.
//
//       k     x       c      =        c
//    +-----+     +----…---+    +----…------+
//    |     |     |        |    |           |
//  n |     |   k |        |  n |           |
//    |     |     |        |    |           |
//    |     |     +----…---+    |           |
//    +-----+                   +----…------+
//
//     xform        input          output
//
// various wrap-around boundaries:
//
// simd_width... we have two full simd registers doing aligned reads,
// but we will have to extract a single simd register worth of data
// from it. We need register pairs for:
//
// * xform stream
// * input stream
//
// We don't need them for subproducts but we need to sum these, so one
// way of keeping dot product components separate is to use a similar
// register pair setup.
//
// kw ... full dot product
//
// complicated a bit because two cases:
//
// a) kw <= simd_width
// b) kw >  simd_width
//
// In the first case, we will get at least one dot product from each
// simd multiply. In the second, we have to do several simd operations
// in order to get a full dot product.
//
// we have efficient ways of summing across vectors by using shifts
// and xor, as opposed to taking n - 1 xor steps to sum n values
//
// nkw ... wrap around transform matrix
//
// if nkw is coprime to simd_width, then we would be heading in
// non-aligned read territory here. Assuming that is the case:
//
// ah, I need two registers for products. 
//
// kwc ... wrap around right of input matrix
//
// this will be coprime to n each time we wrap around, so we always
// restart at a different row.
//

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn all_primes_lcm() {
	assert_eq!(lcm(2,7), 2 * 7);
    }

    #[test]
    fn common_factor_lcm() {
	// 14 = 7 * 2, so 2 is a common factor
	assert_eq!(lcm(2,14), 2 * 7);
    }

    #[test]
    #[should_panic]
    fn zero_zero_lcm() {
	// triggers division by zero (since gcd(0,0) = 0)
	assert_eq!(lcm(0,0), 0 * 0);
    }

    #[test]
    fn one_anything_lcm() {
	assert_eq!(lcm(1,0), 0);
	assert_eq!(lcm(1,1), 1);
	assert_eq!(lcm(1,2), 2);
	assert_eq!(lcm(1,14), 14);
    }

    #[test]
    fn anything_one_lcm() {
	assert_eq!(lcm(0,1), 0);
	assert_eq!(lcm(1,1), 1);
	assert_eq!(lcm(2,1), 2);
	assert_eq!(lcm(14,1), 14);
    }

    #[test]
    fn anything_one_gcd() {
	assert_eq!(gcd(0,1), 1);
	assert_eq!(gcd(1,1), 1);
	assert_eq!(gcd(2,1), 1);
	assert_eq!(gcd(14,1), 1);
    }

    #[test]
    fn one_anything_gcd() {
	assert_eq!(gcd(1,0), 1);
	assert_eq!(gcd(1,1), 1);
	assert_eq!(gcd(1,2), 1);
	assert_eq!(gcd(1,14), 1);
    }

    #[test]
    fn common_factors_gcd() {
	assert_eq!(gcd(2 * 2 * 2 * 3, 2 * 3 * 5), 2 * 3);
	assert_eq!(gcd(2 * 2 * 3 * 3 * 5, 2 * 3 * 5 * 7), 2 * 3 * 5);
    }

    #[test]
    fn coprime_gcd() {
	assert_eq!(gcd(9 * 16, 25 * 49), 1);
	assert_eq!(gcd(2 , 3), 1);
    }

    #[test]
    fn test_lcm3() {
	assert_eq!(lcm3(2*5, 3*5*7, 2*2*3), 2 * 2 * 3 * 5 * 7);
    }

    #[test]
    fn test_lcm4() {
	assert_eq!(lcm4(2*5, 3*5*7, 2*2*3, 2*2*2*3*11),
		   2* 2 * 2 * 3 * 5 * 7 * 11);
    }

    #[test]
    fn test_gcd3() {
	assert_eq!(gcd3(1,3,7), 1);
	assert_eq!(gcd3(2,4,8), 2);
	assert_eq!(gcd3(4,8,16), 4);
	assert_eq!(gcd3(20,40,80), 20);
    }

    #[test]
    fn test_gcd4() {
	assert_eq!(gcd4(1,3,7,9), 1);
	assert_eq!(gcd4(2,4,8,16), 2);
	assert_eq!(gcd4(4,8,16,32), 4);
	assert_eq!(gcd4(20,40,60,1200), 20);
    }

    #[test]
    fn test_macro() {
	new_xform_reader!(the_struct, 3, 4, 1, r0, r1);
	assert_eq!(the_struct.k, 3);
	assert_eq!(the_struct.n, 4);
	assert_eq!(the_struct.w, 1);
	the_struct.xptr += 1;
	assert_eq!(the_struct.xptr, 1);
    }

}
