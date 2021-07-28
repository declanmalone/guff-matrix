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
//! # Building 
//!
//! For x86_64 and Armv8 (Aarch64), building should require no extra
//! options:
//!
//! ```bash
//! cargo build
//! ```
//!
//! # Optional Features
//!
//! Currently available options are:
//! 
//! - **`simulator`** — Software simulation of "wrap-around read matrix"
//!   ("warm") multiply
//! - **`arm_dsp`** — Armv6 (dsp) 4-way SIMD multiply
//! - **`arm_long`** — Armv7/Armv8 8-way NEON reimplementation of Armv6 code
//! - **`arm_vmull`** — Armv7/Armv8 8-way NEON vmull/vtbl multiply
//!
//! To enable building these, use the `--features` option when
//! building, eg:
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --features arm_vmull
//! ```
//!
//! # Software Simulation Feature
//!
//! I've implemented two Rust version of the matrix multiplication
//! code. See the simulator module for details.
//! 
//! The overall organisation of the main functionality of this crate
//! is modelled on the second simulation (SIMD version).
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

use guff::*;

// Only one x86 implementation, included automatically
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

// I want to emit assembly for these
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn _monomorph() {

    use crate::x86::*;

    #[inline(never)]
    fn inner_fn<S : Simd + Copy>(
	xform  : &mut impl SimdMatrix<S>,
	input  : &mut impl SimdMatrix<S>,
	output : &mut impl SimdMatrix<S>) {
	unsafe {
	    simd_warm_multiply(xform, input, output);
	}
    }
    let identity = [
	1,0,0, 0,0,0, 0,0,0,
	0,1,0, 0,0,0, 0,0,0,
	0,0,1, 0,0,0, 0,0,0,
	0,0,0, 1,0,0, 0,0,0,
	0,0,0, 0,1,0, 0,0,0,
	0,0,0, 0,0,1, 0,0,0,
	0,0,0, 0,0,0, 1,0,0,
	0,0,0, 0,0,0, 0,1,0,
	0,0,0, 0,0,0, 0,0,1,
    ];
    let mut transform =	// mut because of iterator
	X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(9,9,true);
    transform.fill(&identity[..]);
    
    // 17 is coprime to 9
    let mut input =
	X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(9,17,false);
    let vec : Vec<u8> = (1u8..=9 * 17).collect();
    input.fill(&vec[..]);

    let mut output =
	X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(9,17,false);

    // works if output is stored in colwise format
    inner_fn(&mut transform, &mut input, &mut output);
    // array has padding, so don't compare that
    assert_eq!(output.array[0..9*17], vec);
}

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

/// Greatest Common Divisor for two integers
pub fn gcd(mut a : usize, mut b : usize) -> usize {
  let mut t;
    loop {
	if b == 0 { return a }
	t = b;
	b = a % b;
	a = t;
    }
}

/// Greatest Common Divisor for three integers
pub fn gcd3(a : usize, b : usize, c: usize) -> usize {
    gcd(a, gcd(b,c))
}

/// Greatest Common Divisor for four integers
pub fn gcd4(a : usize, b : usize, c: usize, d : usize) -> usize {
    gcd(gcd(a,b), gcd(c,d))
}

/// Least Common Multiple for two integers
pub fn lcm(a : usize, b : usize) -> usize {
    (a / gcd(a,b)) * b
}

/// Least Common Multiple for three integers
pub fn lcm3(a : usize, b : usize, c: usize) -> usize {
    lcm( lcm(a,b), c)
}

/// Least Common Multiple for four integers
pub fn lcm4(a : usize, b : usize, c: usize, d : usize) -> usize {
    lcm( lcm(a,b), lcm(c,d) )
}

/// SIMD support, based on `simulator` module
///
/// This trait will be in main module and will have to be implemented
/// for each architecture
pub trait Simd {
    type E : std::fmt::Display;			// elemental type, eg u8
    type V;			// vector type, eg [u8; 8]
    const SIMD_BYTES : usize;

    fn zero_vector() -> Self;

    fn cross_product(a : Self, b : Self) -> Self;
    unsafe fn sum_across_n(m0 : Self, m1 : Self, n : usize, off : usize)
			   -> (Self::E, Self);

    // helper functions for working with elemental types. An
    // alternative to using num_traits.
    fn zero_element() -> Self::E;
    fn add_elements(a : Self::E, b : Self::E) -> Self::E;


    // moved from SimdMatrix
    unsafe fn read_next(mod_index : &mut usize,
			array_index : &mut usize,
			array     : &[Self::E],
			size      : usize,
			ra_size : &mut usize,
			ra : &mut Self)
	-> Self
    where Self : Sized;
    

}

// For the SimdMatrix trait, I'm not going to distinguish between
// rowwise and colwise variants. The iterators will just treat the
// data as a contiguous block of memory. It's only when it comes to
// argument checking (to matrix multiply) and slower get/set methods
// that the layout matters.
//
// Having only one trait also cuts down on duplicated definitions.

// Make it generic on S : Simd, because the iterator returns values of
// that type.

/// Trait for a matrix that supports Simd iteration
pub trait SimdMatrix<S : Simd> {
    // const IS_ROWWISE : bool;
    // fn is_rowwise(&self) -> bool { Self::IS_ROWWISE }

    // size (in bits) of simd vector 
    // const SIMD_SIZE : usize;

    // required methods
    fn is_rowwise(&self) -> bool;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;

    // reset read_next state
    //
    // When called in a loop, input matrices will generally have new
    // data in them, but xform will continue being the same. This
    // means that re-using the xform can/will result in the
    // read_next() state being wrong. It doesn't matter so much for
    // input matrix, since fill() should reset state to zero.
    fn reset(&mut self);

    // Wrap-around read of matrix, returning a Simd vector type
    // 
    // unsafe fn read_next(&mut self) -> S; // moved to Simd
    
    // Wrap-around diagonal write of (output) matrix
    // fn write_next(&mut self, val : S::E); // moved to matrix mul

    

    
    fn indexed_write(&mut self, index : usize, elem : S::E);
    fn as_mut_slice(&mut self) -> &mut [S::E];
    fn as_slice(&self) -> &[S::E];

    // not required by multiply. Maybe move to a separate accessors
    // trait. Comment out for now.
    // fn get(&self, r : usize, c : usize) -> S::E;
    // fn set(&self, r : usize, c : usize, elem : S::E);

    // Convenience stuff
    fn rowcol_to_index(&self, r : usize, c : usize) -> usize {
	// eprintln!("r: {}, c: {}, is_rowwise {}; rows: {}, cols: {}",
	//  r, c, self.is_rowwise(), self.rows(), self.cols() );
	if self.is_rowwise() {
	    r * self.cols() + c
	} else {
	    r + c * self.rows()
	}
    }
    fn size(&self) -> usize { self.rows() * self.cols() }

}


pub unsafe fn simd_warm_multiply<S : Simd + Copy>(
    xform  : &mut impl SimdMatrix<S>,
    input  : &mut impl SimdMatrix<S>,
    output : &mut impl SimdMatrix<S>) {

    // dimension tests
    let c = input.cols();
    let n = xform.rows();
    let k = xform.cols();

    // regular asserts, since they check user-supplied vars
    assert!(k > 0);
    assert!(n > 0);
    assert!(c > 0);
    assert_eq!(input.rows(), k);
    assert_eq!(output.cols(), c);
    assert_eq!(output.rows(), n);

    // searching for prime factors ... needs more work?
    // use debug_assert since division is often costly
    if n > 1 {
	let denominator = gcd(n,c);
	debug_assert_ne!(n, denominator);
	debug_assert_ne!(c, denominator);
    }

    // algorithm not so trivial any more, but still quite simple
    let mut dp_counter  = 0;
    let mut sum         = S::zero_element();
    let simd_width = S::SIMD_BYTES;

    // Code for read_next() that was handled in SimdMatrix has now
    // moved to Simd. We need to track those variables here.
    let mut xform_mod_index = 0;
    let mut xform_array_index = 0;
    let     xform_array = xform.as_slice();
    let     xform_size  = xform.size();
    let mut xform_ra_size = 0;
    let mut xform_ra = S::zero_vector();

    let mut input_mod_index = 0;
    let mut input_array_index = 0;
    let     input_array = input.as_slice();
    let     input_size  = input.size();
    let mut input_ra_size = 0;
    let mut input_ra = S::zero_vector();

    // we handle or and oc (was in matrix class)
    let mut or : usize = 0;
    let mut oc : usize = 0;
    let orows = output.rows();
    let ocols = output.cols();

    // read ahead two products

    let mut i0 : S;
    let mut x0 : S;

    x0 = S::read_next(&mut xform_mod_index,
			 &mut xform_array_index,
			 xform_array,
			 xform_size,
			 &mut xform_ra_size,
			 &mut xform_ra);
    i0 = S::read_next(&mut input_mod_index,
			 &mut input_array_index,
			 input_array,
			 input_size,
			 &mut input_ra_size,
			 &mut input_ra);

    let mut m0 = S::cross_product(x0,i0);

    x0 = S::read_next(&mut xform_mod_index,
			 &mut xform_array_index,
			 xform_array,
			 xform_size,
			 &mut xform_ra_size,
			 &mut xform_ra);
    i0 = S::read_next(&mut input_mod_index,
			 &mut input_array_index,
			 input_array,
			 input_size,
			 &mut input_ra_size,
			 &mut input_ra);
    let mut m1  = S::cross_product(x0,i0);

    let mut offset_mod_simd = 0;
    let mut total_dps = 0;
    let target = n * c;		// number of dot products

    while total_dps < target {

	// at top of loop we should always have m0, m1 full

	// apportion parts of m0,m1 to sum

	// handle case where k >= simd_width
	while dp_counter + simd_width <= k {
	    let (part, new_m)
		= S::sum_across_n(m0,m1,simd_width,offset_mod_simd);
	    sum = S::add_elements(sum,part);
	    m0 = new_m;
	    // x0  = xform.read_next();
	    // i0  = input.read_next();
	    x0 = S::read_next(&mut xform_mod_index,
				 &mut xform_array_index,
				 xform_array,
				 xform_size,
				 &mut xform_ra_size,
				 &mut xform_ra);
	    i0 = S::read_next(&mut input_mod_index,
				 &mut input_array_index,
				 input_array,
				 input_size,
				 &mut input_ra_size,
				 &mut input_ra);
	    m1  = S::cross_product(x0,i0); // new m1
	    dp_counter += simd_width;
	    // offset_mod_simd unchanged
	}
	// above may have set dp_counter to k already.
	if dp_counter < k {	       // If not, ...
	    let want = k - dp_counter; // always strictly positive

	    // eprintln!("Calling sum_across_n with m0 {:?}, m1 {:?}, n {}, offset {}",
	    //      m0.vec, m1.vec, want, offset_mod_simd);
	    let (part, new_m) = S::sum_across_n(m0,m1,want,offset_mod_simd);

	    // eprintln!("got sum {}, new m {:?}", part, new_m.vec);

	    sum = S::add_elements(sum,part);
	    if offset_mod_simd + want >= simd_width {
		// consumed m0 and maybe some of m1 too
		m0 = new_m;	// nothing left in old m0, so m0 <- m1
		// x0  = xform.read_next();
		// i0  = input.read_next();
		x0 = S::read_next(&mut xform_mod_index,
				     &mut xform_array_index,
				     xform_array,
				     xform_size,
				     &mut xform_ra_size,
				     &mut xform_ra);
		i0 = S::read_next(&mut input_mod_index,
				     &mut input_array_index,
				     input_array,
				     input_size,
				     &mut input_ra_size,
				     &mut input_ra);
		m1  = S::cross_product(x0,i0); // new m1
	    } else {
		// got what we needed from m0 but it still has some
		// unused data left in it
		m0 = new_m;
		// no new m1
	    }
	    // offset calculation the same for both arms above
	    offset_mod_simd += want;
	    if offset_mod_simd >= simd_width {
		offset_mod_simd -= simd_width
	    }
	}

	// sum now has a full dot product
	// eprintln!("Sum: {}", sum);

	// handle writing and incrementing or, oc
	let write_index = output.rowcol_to_index(or,oc);
        output.indexed_write(write_index,sum);
	or = if or + 1 < orows { or + 1 } else { 0 };
	oc = if oc + 1 < ocols { oc + 1 } else { 0 };

        sum = S::zero_element();
        dp_counter = 0;
	total_dps += 1;
    }
}


/// Reference matrix multiply. Doesn't use SIMD at all, but uses
/// generic Simd types to be compatible with actual Simd
/// implementations. Note that this multiply routine does not check
/// the gcd condition so it can be used to multiply matrices of
/// arbitrary sizes.
pub fn reference_matrix_multiply<S : Simd + Copy, G>(
    xform  : &mut impl SimdMatrix<S>,
    input  : &mut impl SimdMatrix<S>,
    output : &mut impl SimdMatrix<S>,
    field  : &G)
where G : GaloisField,
<S as Simd>::E: From<<G as GaloisField>::E> + Copy,
<G as GaloisField>::E: From<<S as Simd>::E> + Copy
{

    // dimension tests
    let c = input.cols();
    let n = xform.rows();
    let k = xform.cols();

    // regular asserts, since they check user-supplied vars
    assert!(k > 0);
    assert!(n > 0);
    assert!(c > 0);
    assert_eq!(input.rows(), k);
    assert_eq!(output.cols(), c);
    assert_eq!(output.rows(), n);

    let xform_array  = xform.as_slice();
    let input_array  = input.as_slice();
    // let mut output_array = output.as_mut_slice();

    for row in 0..n {
	for col in 0..c {
	    let xform_index  = xform.rowcol_to_index(row,0);
	    let input_index  = input.rowcol_to_index(0,col);
	    let output_index = output.rowcol_to_index(row,col);

	    let mut dp = S::zero_element();
	    for i in 0..k {
		dp = S::add_elements(dp, field
				     .mul(xform_array[xform_index + i].into(),
					  input_array[input_index + i].into()
				     ).into());
	    }
	    output.indexed_write(output_index,dp);
	}
    }
}

// TODO: make a NoSimd : Simd type and associated matrix types
//
// Right now, the only concrete implementation of these is in the x86
// crate. I should have types available here that can still use
// simd_warm_multiply (with simulated SIMD) or
// reference_matrix_multiply().
//
// AND/OR: an ArchSimd : Simd type and associated matrix types
//
// If the appropriate arch is available and (if needed) one of the
// arch-specific features are enabled, this wrapper layer will call
// them. If they're not, we'll get pure-Rust fallback implementations.



#[cfg(test)]
mod tests {

    use super::*;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use super::x86::*;
    use guff::{GaloisField, new_gf8};

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

    //#[test]

    // fn test_macro() {
    // 	new_xform_reader!(the_struct, 3, 4, 1, r0, r1);
    // 	assert_eq!(the_struct.k, 3);
    // 	assert_eq!(the_struct.n, 4);
    // 	assert_eq!(the_struct.w, 1);
    // 	the_struct.xptr += 1;
    // 	assert_eq!(the_struct.xptr, 1);
    // }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // test taken from simulator.rs
    fn simd_identity_k9_multiply_colwise() {
	unsafe {
	    let identity = [
		1,0,0, 0,0,0, 0,0,0,
		0,1,0, 0,0,0, 0,0,0,
		0,0,1, 0,0,0, 0,0,0,
		0,0,0, 1,0,0, 0,0,0,
		0,0,0, 0,1,0, 0,0,0,
		0,0,0, 0,0,1, 0,0,0,
		0,0,0, 0,0,0, 1,0,0,
		0,0,0, 0,0,0, 0,1,0,
		0,0,0, 0,0,0, 0,0,1,
	    ];
	    let mut transform =	// mut because of iterator
		X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(9,9,true);
	    transform.fill(&identity[..]);

	    // 17 is coprime to 9
	    let mut input =
		X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(9,17,false);
	    let vec : Vec<u8> = (1u8..=9 * 17).collect();
	    input.fill(&vec[..]);
	    
	    let mut output =
		X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(9,17,false);

	    // works if output is stored in colwise format
	    simd_warm_multiply(&mut transform, &mut input, &mut output);
	    // array has padding, so don't compare that
	    assert_eq!(output.array[0..9*17], vec);
	}
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // test taken from simulator.rs
    fn simd_double_identity() {
	// seems like lower half of matrix not being output
	// copy identity matrix down there to test
	unsafe {
	    let double_identity = [
		1,0,0, 0,0,0, 0,0,0,
		0,1,0, 0,0,0, 0,0,0,
		0,0,1, 0,0,0, 0,0,0,
		0,0,0, 1,0,0, 0,0,0,
		0,0,0, 0,1,0, 0,0,0,
		0,0,0, 0,0,1, 0,0,0,
		0,0,0, 0,0,0, 1,0,0,
		0,0,0, 0,0,0, 0,1,0,
		0,0,0, 0,0,0, 0,0,1,
		1,0,0, 0,0,0, 0,0,0,
		0,1,0, 0,0,0, 0,0,0,
		0,0,1, 0,0,0, 0,0,0,
		0,0,0, 1,0,0, 0,0,0,
		0,0,0, 0,1,0, 0,0,0,
		0,0,0, 0,0,1, 0,0,0,
		0,0,0, 0,0,0, 1,0,0,
		0,0,0, 0,0,0, 0,1,0,
		0,0,0, 0,0,0, 0,0,1,
	    ];
	    let mut transform =	// mut because of iterator
		X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(18,9,true);
	    transform.fill(&double_identity[..]);

	    // 17 is coprime to 9
	    let mut input =
		X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(9,17,false);
	    let vec : Vec<u8> = (1u8..=9 * 17).collect();
	    input.fill(&vec[..]);

	    let mut xform_mod_index = 0;
	    let mut xform_array_index = 0;
	    let     xform_array = transform.as_slice();
	    let     xform_size  = transform.size();
	    let mut xform_ra_size = 0;
	    let mut xform_ra = X86u8x16Long0x11b::zero_vector();

	    let mut input_mod_index = 0;
	    let mut input_array_index = 0;
	    let     input_array = input.as_slice();
	    let     input_size  = input.size();
	    let mut input_ra_size = 0;
	    let mut input_ra = X86u8x16Long0x11b::zero_vector();

	    let mut output =
		X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(18,17,true);

	    // works if output is stored in colwise format
	    simd_warm_multiply(&mut transform, &mut input, &mut output);

	    eprintln!("output has size {}", output.size());
	    eprintln!("vec has size {}", vec.len());
	    let output_slice = output.as_slice();
	    let mut chunks = output_slice.chunks(9 * 17);

	    // can't compare with vec without interleaving, but we can
	    // compare halves:
	    let chunk1 = chunks.next();
	    let chunk2 = chunks.next();
	    assert_eq!(chunk1, chunk2);

	    // for (which, chunk) in chunks.enumerate() {
	    // 	eprintln!("chunk {} has size {}", which, chunk.len());
	    // 	assert_ne!(which, 2); // enumerate only 0, 1
	    // 	if which == 1 { assert_eq!(chunk, vec)};
	    // }
	}
    }

    // test conformance with a variety of matrix sizes
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_ref_simd_conformance() {

	let cols = 19;
	for k in 4..9 {
	    for n in 4..17 {
		eprintln!("testing n={}, k={}", n, k);
		unsafe {
		    let mut transform =	// mut because of iterator
			X86SimpleMatrix::<x86::X86u8x16Long0x11b>
			::new(n,k,true);
		    let mut input =
			X86SimpleMatrix::<x86::X86u8x16Long0x11b>
			::new(k,cols,false);

		    transform.fill(&(1u8..).take(n*k).collect::<Vec<u8>>()[..]);
		    input.fill(&(1u8..).take(k*cols).collect::<Vec<u8>>()[..]);

		    let mut xform_mod_index = 0;
		    let mut xform_array_index = 0;
		    let     xform_array = transform.as_slice();
		    let     xform_size  = transform.size();
		    let mut xform_ra_size = 0;
		    let mut xform_ra = X86u8x16Long0x11b::zero_vector();

		    let mut input_mod_index = 0;
		    let mut input_array_index = 0;
		    let     input_array = input.as_slice();
		    let     input_size  = input.size();
		    let mut input_ra_size = 0;
		    let mut input_ra = X86u8x16Long0x11b::zero_vector();


		    let mut ref_output =
			X86SimpleMatrix::<x86::X86u8x16Long0x11b>
			::new(n,cols,true);

		    let mut simd_output =
			X86SimpleMatrix::<x86::X86u8x16Long0x11b>
			::new(n,cols,true);

		    // do multiply both ways
		    simd_warm_multiply(&mut transform, &mut input,
				       &mut simd_output);
		    reference_matrix_multiply(&mut transform,
					      &mut input,
					      &mut ref_output,
					      &new_gf8(0x11b, 0x1b));

		    assert_eq!(format!("{:x?}", ref_output.as_slice()),
			       format!("{:x?}", simd_output.as_slice()));
		}
	    }
	}
    }
}
