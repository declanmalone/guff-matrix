//#![feature(stdsimd)]


#[cfg(target_arch = "arm")]
use core::arch::arm::*;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use std::mem::transmute;


// Quick notes on approaching this ...
//
// I'm going to shift all simd-related logic out of the matrix
// implementation in x86, and will follow suit here. It will only have
// stuff relating to storage, and so can be generic.
//
// I want to avoid code duplication among the two or three Arm
// implementations, so I might be able to split the Simd trait up into
// separate subtraits and put the stuff that's common to them into
// one, with stuff that's different (like the SIMD multiply routine)
// into another.
//
// I'm leaning towards doing more of the work in the matrix multiply.
// The matrix implementation will no longer track readahead, so the
// Simd objects will have to be able to provide support for
// translating between high-ish level concepts that the matrix
// multiply routine will use (like combining a new SIMD worth of data
// with the old readahead) and how they're actually implemented in the
// hardware (eg, memory reads, using stored or calculated masks, and
// combination of shifts/rotates/extracts/whatever)
//
// Comparing x86 to Arm, neither has variable shifts or
// rotates. They're all done by constant amounts. On x86, the best way
// to handle most of the work is by using pshufb to emulate shifts.
//
// On the Arm side, it's maybe a bit easier to emulate rotate:
//
// Start with an initial rotate mask of 0..15 (or 0..7 for 64-bit
// vectors)
//
// doing a vtbl using the mask on the data to be rotated initially
// gives the same data back.
//
// Adding 1 to each lane and anding with 15 (or 7) ends up with all
// the values 0..15 rotated by 1.
//
// Rotates are handy for dealing with masks, too. If we have a product
// stream and we want to take k values out of it each time (or some
// other constant, if k > simd width), then we can set up a starting
// mask with k zeros and simd - k ones (0xff). We can use that ANDed
// with the data to only get the values we want. Then we can rotate
// the mask by k (or the other constant) to get the next values.
//
// Speaking of 64-bit vectors, this routine here uses them because
// there isn't a 128-wide table lookup on armv7 NEON.
//
// Anyway, I might implement another version of this for armv8, which
// has the required instructions, I think.
//
// The two architectures (x86 and Arm) are sufficiently different that
// it might make sense to use rotate-based masking extensively on the
// Arm side, and shift-based operations on x86. This means that it
// might be a bit difficult to have a "generic" interface between the
// matrix multiply and Simd code. However, as a temporary measure, I
// can implement *both* ways of doing the same thing in the matrix
// code, but only code the implementations that make sense in the
// target Simd architecture, with the other calls becoming no-ops.
//

// Since there's a fair bit of refactoring needed to get Arm and x86
// working harmoniously together (in terms of traits and
// implementations), not to mention new matrix code and all the
// boilerplate needed to implement the existing trait setup, I'll take
// a different approach. I'll work from the bottom up, implementing
// rotates (and maybe shifts) and general wrap-around reading (as
// opposed to being tied to a matrix), test them, and then write a
// separate matrix multiply routine (and matrix-handling code, too,
// probably).
//
// Then, once I have this Arm-specific implementation, I can start
// looking at reworking the original Simd/Matrix/Multiply combination
// to follow the Arm version.
//
// Fortunately, the current version of the code still compiles on Arm,
// although it doesn't do anything useful since there's no Matrix
// implementation. The benchmarks don't work, though, but that's not a
// problem right now.
// 
// I'll be focusing on the vmull/tbl version (this file) first among
// the various Arm implementations since it works on all my (non-Pi)
// boards and it's got the best performance.

pub trait ArmSimd {

    type V;			// main vector storage type
    type E;			// element type (u8)
    const SIMD_BYTES : usize;	// simd width in bytes

    // will have extra types here describing masks and such that will
    // be needed by the matrix multiply routine to track state

    // required fns will be named after the operations as the matrix
    // multiply fn sees the task at hand, eg:

    //      unsafe fn non_wrapping_read(ptr : *Self::E) -> Self;

    // depending on the level of abstraction, I might have a
    // wrapping_read() fn for reading past end of buffer, or I might
    // break it down into smaller parts like read_remainder() or
    // partial_read() and combine_remainder_and_new_stream()... or
    // something like that. I could express it in terms of updates to
    // readahead.

    // One thing that I should be testing fairly early on: the simd
    // multiply.
    //
    // If I implement my matrices like I did in the simulator (ie,
    // implementing Iterator<SimdType>), then I can pretty quickly
    // code up a matrix multiply routine. Although all the iterator
    // state will still be in the matrix type.

    // Also, for clarity, I could also wrap some of this stuff using
    // Option. For example, if I pass in the array bounds to
    // non_wrapping_read(), then it could return Option<Self> which
    // could be None to indicate that the read would wrap. Again,
    // though, my aim is to move the logic into the matrix multiply
    // routine, not hide it from it... still, that approach does have
    // its merits, even if it's only for prototyping.

    // In a similar vein, we can encapsulate reader state a lot more
    // cleanly if we give it its own struct. Again, using Option, we
    // can have add_partial (for consuming bytes at the end of the
    // matrix) and add_full (adding a full simd read), which will
    // return Some(Simd) if there are enough bytes in it, otherwise it
    // will return None. It's not much of a bother to "inline" the
    // logic into the matrix multiply later, or maybe we can even keep
    // it in the final program (I dislike having to unwrap(), though,
    // so I could rewrite it as add_partial_expecting_none() and
    // add_partial_expecting_some() to avoid duplicate caller/callee
    // checks, and use debug_assert!() to enforce the calling logic)

    // Yes... actually separate structs are a good thing. It might
    // mean copying a bit more data (if the compiler can't figure out
    // that some struct members are unchanged, which I don't think it
    // can), but it's a much better way to code and test...

    // Anyway, let's go with the idea of Option wrapping, anyway

    // Need to think about boundary condition... should it be on
    // beyond, or on the end of the matrix? Just be consistent.
    
    unsafe fn non_wrapping_read(read_ptr :  const* Self::E,
				beyond   :  const* Self::E
    ) -> Option<Self>; // None if read_ptr + SIMD_BYTES >= beyond

    // And a version that wraps around matrix boundary
    unsafe fn wrapping_read(read_ptr : const* Self::E,
			    beyond   : const* Self::E,
			    restart  : const* Self::E
    ) -> (Self, Option<Self>); // if non-wrapping fails

    // 

}

// Q: why did I stop using Iterator to implement read_next?
//
// I think that the reason is that I saw that it wasn't inlining
// properly, even with the inline directive. 

pub struct VmullEngine8x8 {
    vec : uint8x8_t,
}

// low-level intrinsics
impl VmullEngine8x8 {

    unsafe fn read_simd_poly(ptr: const* u8) -> poly8x8_t {
	vld1_p8(ptr)
    }

    unsafe fn read_simd_uint(ptr: const* u8) -> uint8x8_t {
	vld1_u8(ptr)
    }

    unsafe fn xor_across(v : Self) -> u8 {
	
    }
}

// Type conversion is very useful
impl From<uint8x8_t> for VmullEngine8x8 {
    fn from(other : uint8x8_t) {
	unsafe {
	    Self { vec : other }
	}
    }
}
impl From<poly8x8_t> for VmullEngine8x8 {
    fn from(other : poly8x8_t) {
	unsafe {
	    Self { vec : vreinterpretq_u8_p8(other) }
	}
    }
}

// Can rust automatically derive Into for the above? I guess so ... it
// should just translate other.into() as self.from().

// Do I want a bunch more foreign implementations on wide forms, eg:
//
// (actually, not allowed unless I make my own types; not sure if type
// aliases count, but I'm not going to bother trying that now)
//
// impl From<uint16x8_t> for poly16x8_t {
//    fn from(other : uint16x8_t) {
// 	unsafe {
// 	    vreinterpretq_p16_u16(other)
// 	}
//     }
// }
// impl From<poly16x8_t> for uint16x8_t {
//    fn from(other : poly16x8_t) {
// 	unsafe {
// 	    vreinterpretq_u16_p16(other)
// 	}
//     }
// }


impl ArmSimd for VmullEngine8x8 {
    type V = poly8x8_t;
    type E = u8;

    

}



// Interleaving C version in comments

// void simd_mull_reduce_poly8x8(poly8x8_t *result,
//			      poly8x8_t *a, poly8x8_t *b) {

// TODO: make this (or a wrapping function) return a poly8x8_t
pub fn simd_mull_reduce_poly8x8(result : *mut u8,
			 a : &poly8x8_t, b: &poly8x8_t) {

    unsafe {
	// // do non-modular poly multiply
	// poly16x8_t working = vmull_p8(*a,*b);
	let mut working : poly16x8_t = vmull_p8(*a, *b);

	// // copy result, and shift right
	// uint16x8_t top_nibble = vshrq_n_u16 ((uint16x8_t) working, 12);
	let mut top_nibble : uint16x8_t = vshrq_n_u16 (vreinterpretq_u16_p16(working), 12);

	//  // was uint8x16_t, but vtbl 
	//  static uint8x8x2_t u4_0x11b_mod_table =  {
	//    0x00, 0x1b, 0x36, 0x2d, 0x6c, 0x77, 0x5a, 0x41,
	//    0xd8, 0xc3, 0xee, 0xf5, 0xb4, 0xaf, 0x82, 0x99,
	//  };

	// shift table for poly 0x11b
	let tbl_1 : uint8x8_t = transmute([0x00u8, 0x1b, 0x36, 0x2d, 0x6c, 0x77, 0x5a, 0x41, ]);
	let tbl_2 : uint8x8_t = transmute([0xd8u8, 0xc3, 0xee, 0xf5, 0xb4, 0xaf, 0x82, 0x99, ]);
	let u4_0x11b_mod_table = uint8x8x2_t ( tbl_1, tbl_2 );

	// looks like we can't get a uint16x8_t output, so have to break up
	// into two 8x8 lookups. Can we cast to access the halves?

	//   uint8x8_t reduced = vmovn_u16(top_nibble);
	let mut reduced : uint8x8_t = vmovn_u16(top_nibble);

	// now we should have what we need to do 8x8 table lookups
	//  uint8x8_t lut = vtbl2_u8(u4_0x11b_mod_table, reduced);
	let mut lut : uint8x8_t = vtbl2_u8(u4_0x11b_mod_table, reduced);

	// Next, have to convert u8 to u16, shifting left 4 bits
	//  poly16x8_t widened = (poly16x8_t) vmovl_u8(lut);

	// try out foreign from/into: (ah, doesn't work; I'd have to
	// wrap foreign types in my own newtype)
	//
	// let mut widened : poly16x8_t = (vmovl_u8(lut)).into();
	//
	
	let mut widened : poly16x8_t = vreinterpretq_p16_u16(vmovl_u8(lut));

	// uint16x8_t vshlq_n_u16 (uint16x8_t, const int)
	// Form of expected instruction(s): vshl.i16 q0, q0, #0
	//  widened = (poly16x8_t) vshlq_n_u16((uint16x8_t) widened, 4);
	widened = vreinterpretq_p16_u16(vshlq_n_u16(vreinterpretq_u16_p16(widened), 4));


	// uint16x8_t veorqq_u16 (uint16x8_t, uint16x8_t)
	// Form of expected instruction(s): veorq q0, q0, q0
	//  working = (poly16x8_t) veorq_u16((uint16x8_t) working, (uint16x8_t) widened);
	working = vreinterpretq_p16_u16(veorq_u16(
	    vreinterpretq_u16_p16(working),
	    vreinterpretq_u16_p16(widened)));

	// First LUT complete... repeat steps
  
	// extra step to clear top nibble to get at the one to its right
	//  top_nibble = vshlq_n_u16 ((uint16x8_t) working, 4);
	top_nibble = vshlq_n_u16 (vreinterpretq_u16_p16(working), 4);

	// Now just copy/paste other steps
	//  top_nibble = vshrq_n_u16 ((uint16x8_t) top_nibble, 12);
	top_nibble = vshrq_n_u16 (top_nibble, 12);
	//  reduced = vmovn_u16(top_nibble);
	reduced = vmovn_u16(top_nibble);
	//  lut = vtbl2_u8(u4_0x11b_mod_table, reduced);
	lut = vtbl2_u8(u4_0x11b_mod_table, reduced);
	//  widened = (poly16x8_t) vmovl_u8(lut);
	widened = vreinterpretq_p16_u16(vmovl_u8(lut));
	// remove step, since we're applying to low byte
	// // widened = (poly16x8_t) vshlq_n_u16((uint16x8_t) widened, 4);
	
	// working = (poly16x8_t) veorq_u16((uint16x8_t) working, (uint16x8_t) widened);
	working = vreinterpretq_p16_u16(veorq_u16(
	    vreinterpretq_u16_p16(working),
	    vreinterpretq_u16_p16(widened)));

	// apply mask (vand expects 2 registers, so use shl, shr combo)
	//  working = (poly16x8_t) vshlq_n_u16 ((uint16x8_t) working, 8);
	//  working = (poly16x8_t) vshrq_n_u16 ((uint16x8_t) working, 8);
	working = vreinterpretq_p16_u16(vshlq_n_u16 (vreinterpretq_u16_p16(working), 8));
	working = vreinterpretq_p16_u16(vshrq_n_u16 (vreinterpretq_u16_p16(working), 8));

	// use narrowing mov to send back result
	//  *result = (poly8x8_t) vmovn_u16((uint16x8_t) working);
	let narrowed : uint8x8_t = vmovn_u16(vreinterpretq_u16_p16(working));
	vst1_u8(result, narrowed);	
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    use guff::{GaloisField,new_gf8};

    #[test]
    fn test_mull_reduce_poly8x8() {
	let mut fails = 0;
	let a_array = [0u8,10,20,30,40,50,60,70];
	let b_array = [8u8,9,10,11,12,13,14,15];
	let a : poly8x8_t;
	let b : poly8x8_t;
	unsafe {
	    a = transmute ( a_array );
	    b = transmute ( b_array );
	}
	let mut r : poly8x8_t;

	let mut result : Vec<u8> = vec![0;8];

	let f = new_gf8(0x11b, 0x1b);
	simd_mull_reduce_poly8x8(result.as_mut_ptr(), &a, &b);

	for i in 0 .. 8 {
	    let got    = result[i];
	    let expect = f.mul(a_array[i], b_array[i]);
	    assert_eq!(got, expect);
	}
    }
}
