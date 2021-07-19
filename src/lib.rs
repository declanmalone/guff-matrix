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

// SIMD support, based on `simulator` module

// This trait will be in main module and will have to be implemented
// for each architecture
pub trait Simd {
    type E : std::fmt::Display;			// elemental type, eg u8
    type V;			// vector type, eg [u8; 8]
    const SIMD_BYTES : usize;

    fn cross_product(a : Self, b : Self) -> Self;
    unsafe fn sum_across_n(m0 : Self, m1 : Self, n : usize, off : usize)
			   -> (Self::E, Self);

    // helper functions for working with elemental types. An
    // alternative to using num_traits.
    fn zero_element() -> Self::E;
    fn add_elements(a : Self::E, b : Self::E) -> Self::E;
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
pub trait SimdMatrix<S : Simd> {
    // const IS_ROWWISE : bool;
    // fn is_rowwise(&self) -> bool { Self::IS_ROWWISE }

    // size (in bits) of simd vector 
    const SIMD_SIZE : usize;

    // required methods
    fn is_rowwise(&self) -> bool;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    unsafe fn read_next(&mut self) -> S;
    fn write_next(&mut self, val : S::E);

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
    if k != 1 { debug_assert_ne!(k, gcd(k,c)) }
    
    // algorithm not so trivial any more, but still quite simple
    let mut dp_counter  = 0;
    let mut sum         = S::zero_element();
    let simd_width = S::SIMD_BYTES;
    
    // we don't have mstream any more since we handle it ourselves

    // read ahead two products
    let mut i0 : S;
    let mut x0 : S;

    x0 = xform.read_next();
    i0 = input.read_next();
    let mut m0 = S::cross_product(x0,i0);

    x0  = xform.read_next();
    i0  = input.read_next();
    let mut m1  = S::cross_product(x0,i0);

    let mut offset_mod_simd = 0;
    let mut total_dps = 0;
    let target = n * k * c;
    
    while total_dps < target {

	// at top of loop we should always have m0, m1 full

	// apportion parts of m0,m1 to sum

	// handle case where k >= simd_width
	while dp_counter + simd_width <= k {
	    let (part, new_m)
		= S::sum_across_n(m0,m1,simd_width,offset_mod_simd);
	    sum = S::add_elements(sum,part);
	    m0 = new_m;
	    x0  = xform.read_next();
	    i0  = input.read_next();
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
		x0  = xform.read_next();
		i0  = input.read_next();
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
        output.write_next(sum);
        sum = S::zero_element();
        dp_counter = 0;
	total_dps += 1;
    }
}



#[cfg(test)]
mod tests {

    use super::*;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use super::x86::*;
    
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
}
