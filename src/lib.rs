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
//! to detect the availability of `target_feature = "neon"`. As a
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
pub mod x86;

// Implementation (1) above
#[cfg(all(target_arch = "arm", feature = "arm_dsp"))]
pub mod arm_dsp;

// Implementation (2) above
#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_long"))]
pub mod arm_long;

// Implementation (3) above
#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_vmull"))]
pub mod arm_vmull;

#[cfg(feature = "simulator")]
pub mod simulator;

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


// One approach: trait providing matrix multiply over infinite product
// stream/tape. Then implement the missing bits in each SIMD module.
//
trait StreamingMatrixMul {

    type Elem : std::ops::BitXor<Output=Self::Elem>; // eg, u8
    type SIMD;			// eg, __m128i

    // Make n, c, k and w associated constants.

    // The reason for this is that eventually I might want to have a
    // derive macro that builds (derives) a matrix solver given the
    // appropriate types/constants
    
    const N : usize;
    const C : usize;
    const K : usize;
    const W : usize;
    const SIMD_SIZE : usize;	// length of SIMD / length of Elem
    const DP_FINAL : usize;	// 

    // return zero of appropriate type
    fn zero_product(&self) -> Self::Elem;
    // sum across full SIMD vector
    fn sum_across(&self, v : Self::SIMD) -> Self::Elem;
    // sum across k % simd_size remainder
    fn sum_across_remaining(&self, v : Self::SIMD) -> Self::Elem;

    // these eventually read from xform/input matrices
    fn get_simd_products(&self) -> Self::SIMD;
    fn get_remaining_products(&self) -> Self::SIMD;

    // this eventually writes to output matrix
    fn write_next(&self, elem : Self::Elem);

    fn multiply(&self) {
	let mut product = self.zero_product();
	let mut dp_remaining = Self::K;
	let mut written : usize = 0;
	loop {
	    // if k >= simd_size, add simd_size products at a time
	    while dp_remaining >= Self::SIMD_SIZE {
		product = product ^ self.sum_across(self.get_simd_products());
		dp_remaining -= Self::SIMD_SIZE
	    }
	    // todo: what if we had (simd_size / 2)-way simd engine too?

	    // the remainder will always be k % simd_size
	    if dp_remaining != 0 {
		product = product ^ self.sum_across_remaining(self.get_remaining_products())
	    }
	    self.write_next(product);
	    written += 1;
	    if written == Self::N * Self::C { break }
	    product = self.zero_product();
	    dp_remaining = Self::K;
	}
    }
}

// The above could also be written to use methods to determine the
// constants. The types would still have to be associated types unless
// we rewrote the above method signatures to return no data apart from
// maybe a bool, eg:
//
// if self.is_kw_gt_simd_size() { self.add_full_simd_products() }
// if self.is_kw_mod_simd_size_ne_zero() { self.add_remainder() }
//

// What will implement this? A multiply stream.

// That, in turn, will have to:
//
// Store a transform matrix that implements wrap-around read trait
// Store an input matrix that also implements wrap-around read
// Store an output matrix that implements diagonal write
// Interact with a SIMD engine
//
// I think that I'll follow much the same idea as in the simulator
// except that instead of infinite iterator for reads, we work with
// SIMD-sized chunks.

trait WarmSimd {

    type Elem;
    type SIMD;

    fn read_next_simd(&self) -> Self::SIMD;
}

// Optimising for special cases:
//
// * xform matrix fits into a small number of SIMD registers
// * xform is a multiple of SIMD size
//
// In the first case, we can read the matrix into a register or
// register pair and only use rotates.
//
// In the second case, we can have simplified wrap-around code because
// there will be no need for any shifts/rotates.
//
// The logical conclusion of this sort of optimisation might be to
// work on smaller submatrices. 

// Example of first optimisation
struct SubSimdMatrixStreamer {
    buffer : [u8;8],

    // "rotate" as a name isn't quite right, but general idea is that
    // if we have at least one copy of the matrix in the register we
    // can use shifts and ors to advance the "read pointer" by an
    // arbitrary amount. For example, if simd size is 8, and matrix is
    // 6, we start with the following indexes into the matrix:
    //
    // [0, 1, 2, 3, 4, 5, 0, 1 ]
    //
    // advancing by 8, we want to start at 2 and get:
    //
    // [2, 3, 4, 5, 0, 1, 2, 3 ]
    //
    // We do so by a combination of shl and shr:
    //
    // (buffer << 2) | (buffer >> 4)
    
    shl : usize,
    shr : usize,
}

impl SubSimdMatrixStreamer {

    // constructor will read from memory and write at least one full
    // copy and possibly a partial copy into buffer

    //    fn new() -> Self { }
}

// Now we can make this struct work as a WarmSimd:
impl WarmSimd for SubSimdMatrixStreamer {
    type Elem = u8;
    type SIMD = [u8;8];		// in reality, needs to be generic
    fn read_next_simd(&self) -> Self::SIMD {
	let result = self.buffer;
	// following won't compile yet
	//	self.buffer = (result << self.shl) | (result >> self.shr);
	result
    }

}

// We can have other structures that also implement WarmSimd:
//
// * variants of SubSimdMatrixStreamer, where the array fits into two,
//   three or more SIMD registers (I'm assuming that even though these
//   values are stored in a struct, when they come to be used by
//   StreamingMatrixMul, if we inline all the functions, they'll get
//   loaded once into registers and not be written back out until the
//   multiply is done; not 100% sure that this is so, though)
//
// * more normal case where matrix does not fit in registers, so does
//   actual wrap-around read on memory.
//


// Another approach ...
//
// More of a bottom-up approach: Make some new types representing
// generic 128-bit and 64-bit SIMD types.
//
// Don't have to support every single type of operation... just enough
// to implement buffering and wrap-around reads.
//






//
// * variant where k % simd_size = 0, so remainder function does
//   nothing (actually applies to StreamingMatrixMul, not WarmSimd)
// 


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
