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
    
    unsafe fn non_wrapping_read(read_ptr :  *const Self::E,
				beyond   :  *const Self::E
    ) -> Option<Self>    // None if read_ptr + SIMD_BYTES >= beyond
	where Self : Sized;

    // And a version that wraps around matrix boundary
    unsafe fn wrapping_read(read_ptr : *const Self::E,
			    beyond   : *const Self::E,
			    restart  : *const Self::E
    ) -> (Self, Option<Self>)
    where Self : Sized;  // if non-wrapping fails

    // 

}

// Q: why did I stop using Iterator to implement read_next?
//
// I think that the reason is that I saw that it wasn't inlining
// properly, even with the inline directive. 

#[derive(Debug)]
pub struct VmullEngine8x8 {
    // using uint8x8_t rather than poly8x8_t since it involves less
    // type conversion.
    vec : uint8x8_t,
}

// low-level intrinsics
impl VmullEngine8x8 {

    unsafe fn read_simd(ptr: *const u8) -> Self {
	vld1_p8(ptr).into()
    }

    // unsafe fn read_simd_uint(ptr: *const u8) -> uint8x8_t {
    // 	vld1_u8(ptr)
    // }

    unsafe fn xor_across(v : Self) -> u8 {
	let mut v : uint64x1_t = vreinterpret_u64_u8(v.vec);
	// it seems that n is bits? No. Bytes.
	v = veor_u64(v, vshl_n_u64::<4>(v));
	v = veor_u64(v, vshl_n_u64::<2>(v));
	v = veor_u64(v, vshl_n_u64::<1>(v));
	vget_lane_u8::<0>(vreinterpret_u8_u64(v))
    }

    unsafe fn rotate_right(v : Self, amount : usize) -> Self {
	let mut mask = transmute( [0u8,1,2,3,4,5,6,7] ); // null rotate mask
	let add_amount = vmov_n_u8(amount as u8);
	let range_mask = vmov_n_u8(0x07);
	mask = vadd_u8(mask, add_amount);
	mask = vand_u8(mask, range_mask);
	vtbl1_u8(v.vec, mask).into()
    }

    unsafe fn rotate_left(v : Self, amount : usize) -> Self {
	Self::rotate_right(v, 8 - amount)
    }

    // shift can take +ve numbers (shift right) or -ve (shift left)
    unsafe fn shift(v : Self, amount : isize) -> Self {
	let mut mask = transmute( [0u8,1,2,3,4,5,6,7] ); // null shift mask
	let add_amount = vmov_n_s8(amount as i8);
	// let range_mask = vmov_n_u8(0x07);
	mask = vadd_s8(mask, add_amount);
	// mask = vand_u8(mask, range_mask);
	vreinterpret_u8_s8(vtbl1_s8(vreinterpret_s8_u8(v.vec), mask))
	    .into()
    }

    unsafe fn shift_left(v : Self, amount : usize) -> Self {
	Self::shift(v, -(amount as isize))
    }

    unsafe fn shift_right(v : Self, amount : usize) -> Self {
	Self::shift(v, amount as isize)
    }

    // lo is current time, hi is in the future
    // extracts 8 bytes. Do I need extract_n_from_offset? Maybe.
    unsafe fn extract_from_offset(lo: Self, hi : Self, offset : usize)
				  -> Self {
	debug_assert!(offset < 8);
	let tbl2 = uint8x8x2_t ( lo.vec, hi.vec );
	let mut mask = transmute( [0u8,1,2,3,4,5,6,7] ); // null rotate mask
	let add_amount = vmov_n_u8(offset as u8);
	mask = vadd_u8(mask, add_amount);
	vtbl2_u8(tbl2, mask).into()	
    }

    unsafe fn splat(elem : u8) -> Self {
	vmov_n_u8(elem).into()
    }
    
    unsafe fn mask_start_elements(v : Self, count : usize) -> Self {
	debug_assert!(count > 0);
	let mask = Self::shift_right(Self::splat(0xff),
				     (8usize - count).into());
	vand_u8(v.vec, mask.vec).into()	
    }
    
}

// Type conversion is very useful
impl From<uint8x8_t> for VmullEngine8x8 {
    fn from(other : uint8x8_t) -> Self {
	unsafe {
	    Self { vec : other }
	}
    }
}
impl From<poly8x8_t> for VmullEngine8x8 {
    fn from(other : poly8x8_t) -> Self {
	unsafe {
	    Self { vec : vreinterpret_u8_p8(other) }
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
    type V = uint8x8_t;
    type E = u8;
    const SIMD_BYTES : usize = 8;

    // caller is responsible for tracking read_ptr

    unsafe fn non_wrapping_read(read_ptr :  *const Self::E,
				beyond   :  *const Self::E
    ) -> Option<Self> {
	if read_ptr.offset(Self::SIMD_BYTES as isize) > beyond {
	    None
	} else {
	    Some(Self::read_simd(read_ptr).into())
	}
    }

    unsafe fn wrapping_read(read_ptr : *const Self::E,
			    beyond   : *const Self::E,
			    restart  : *const Self::E
    ) -> (Self, Option<Self>) {

	let missing : isize
	    = (read_ptr.offset(Self::SIMD_BYTES as isize)).offset_from(beyond);
	debug_assert!(missing > 0);

	// get 8 - missing from end
	let mut r0 = Self::read_simd(read_ptr);

	if missing == 0 {
	    return (r0.into(), None);
	}

	// get missing from start
	let mut r1 = Self::read_simd(restart);

	// Two steps to combine...
	// * rotate r0 left by missing (move bytes to top)
	// * extract 8 bytes from {r1:r0} at offset (missing -8)
	r0 = Self::rotate_left(r0.into(), missing as usize);
	r0 = Self::extract_from_offset(r0, r1, missing as usize - 8);

	// To get updated r1 (with missing bytes removed), either:
	// * right shift by missing
	// * mask out lower missing bytes

	// The first places the result in the low bytes of r1, while
	// the second leaves them in the high bytes.

	// Since later on, when we combine readahead (which is what r1
	// is) with the next simd vector, we'll need to shift/rotate
	// it left again. So it would be preferable to only mask the
	// bytes.

	// OTOH, it's easier to implement shifts and rotates using tbl
	// than it is to work with masks (there's apparently no way to
	// create a mask vector from a u8/u16, so we'd need to load up
	// a mask of the appropriate size from memory and then
	// (sometimes) rotate it).
	//
	// I'll pause here and implement basic shifts, rotates and
	// masks, then test them.
	//
	// Actually, second thoughts on masks ... it's actually not so
	// difficult:
	// * splat 0xff to all lanes
	// * shift left or right using:
	// ** [0..simd_width - 1] as initial mask
	// ** add desired shift to all lanes (+ve: shift right)
	// ** use vtbl1 to shift
	//
	// With this, it's easy enough to select or blank (with
	// negated mask) a number of bytes at the start/end.

	(r0, None)
    }

}



// Interleaving C version in comments

// void simd_mull_reduce_poly8x8(poly8x8_t *result,
//			      poly8x8_t *a, poly8x8_t *b) {

// TODO: make this (or a wrapping function) return a poly8x8_t
pub fn simd_mull_reduce_poly8x8(a : &poly8x8_t, b: &poly8x8_t)
				-> poly8x8_t {

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
	//	vst1_u8(result, narrowed);
	vreinterpret_p8_u8(narrowed)
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    use guff::{GaloisField,new_gf8};

    #[test]
    fn test_mull_reduce_poly8x8() {
	// let mut fails = 0;
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
	let got_poly = simd_mull_reduce_poly8x8(&a, &b);
	unsafe {
	    vst1_p8(result.as_mut_ptr(), got_poly);
	}
	for i in 0 .. 8 {
	    let got    = result[i];
	    let expect = f.mul(a_array[i], b_array[i]);
	    assert_eq!(got, expect);
	}
    }

    // Test rotates
    #[test]
    fn test_rotate_right_1() {
	unsafe {
	    let data   : uint8x8_t = transmute([1u8,10,20,30,40,50,60,70]);
	    let expect : uint8x8_t = transmute([10u8,20,30,40,50,60,70,1]);

	    let got = VmullEngine8x8::rotate_right(data.into(), 1);
	    assert_eq!(format!("{:x?}", expect),
		       format!("{:x?}", got.vec));
	}
    }

    #[test]
    fn test_rotate_left_1() {
	unsafe {
	    let data   : uint8x8_t = transmute([1u8,10,20,30,40,50,60,70]);
	    let expect : uint8x8_t = transmute([70u8,1,10,20,30,40,50,60]);

	    let got = VmullEngine8x8::rotate_left(data.into(), 1);
	    assert_eq!(format!("{:x?}", expect),
		       format!("{:x?}", got.vec));
	}
    }


    // Test shifts
    #[test]
    fn test_shift_right_1() {
	unsafe {
	    let data   : uint8x8_t = transmute([1u8,10,20,30,40,50,60,70]);
	    let expect : uint8x8_t = transmute([10u8,20,30,40,50,60,70,0]);

	    let got = VmullEngine8x8::shift_right(data.into(), 1);
	    assert_eq!(format!("{:x?}", expect),
		       format!("{:x?}", got.vec));
	}
    }

    #[test]
    fn test_shift_left_1() {
	unsafe {
	    let data   : uint8x8_t = transmute([1u8, 10,20,30,40,50,60,70]);
	    let expect : uint8x8_t = transmute([0u8, 1,10,20,30,40,50,60]);

	    let got = VmullEngine8x8::shift_left(data.into(), 1);
	    assert_eq!(format!("{:x?}", expect),
		       format!("{:x?}", got.vec));
	}
    }

    // XOR across
    #[test]
    fn test_xor_across() {
	unsafe {
	    let data   : uint8x8_t = transmute([1u8, 2,4,8,16,32,64,128]);
	    let got = VmullEngine8x8::xor_across(data.into());

	    assert_eq!(255, got);
	}
    }

    // extract_from_offset

    #[test]
    fn test_extract_from_offset() {
	unsafe {
	    let r0 : uint8x8_t = transmute([1u8, 2,4,8,16,32,64,128]);
	    let r1 : uint8x8_t = transmute([1u8, 2,3,4,5,6,7,8]);

	    // expected results
	    let off_1 : uint8x8_t = transmute([2u8,4,8,16,32,64,128,1]);

	    let res = VmullEngine8x8::extract_from_offset(r0.into(), r1.into(), 0);
	    assert_eq!(format!("{:x?}", r0),
		       format!("{:x?}", res.vec));

	    let res = VmullEngine8x8::extract_from_offset(r0.into(), r1.into(), 1);
	    assert_eq!(format!("{:x?}", off_1),
		       format!("{:x?}", res.vec));
	}
    }

    #[test]
    fn test_splat() {
	unsafe {
	    let expect : uint8x8_t = transmute([42u8,42,42,42, 42,42,42,42]);
	    let got = VmullEngine8x8::splat(42);
	    assert_eq!(format!("{:x?}", expect),
		       format!("{:x?}", got.vec));
	}
    }

    #[test]
    fn test_mask_start_elements() {
	unsafe {
	    let input : uint8x8_t = transmute([42u8,42,42,42, 42,42,42,42]);
	    let expect_1 : uint8x8_t = transmute([42u8,0 ,0 ,0 , 0 ,0 ,0 ,0 ]);
	    let expect_2 : uint8x8_t = transmute([42u8,42,0 ,0 , 0 ,0 ,0 ,0 ]);
	    let expect_3 : uint8x8_t = transmute([42u8,42,42,0 , 0 ,0 ,0 ,0 ]);
	    let expect_7 : uint8x8_t = transmute([42u8,42,42,42, 42,42,42,0 ]);
	    let expect_8 : uint8x8_t = transmute([42u8,42,42,42, 42,42,42,42]);

	    let got = VmullEngine8x8::mask_start_elements(input.into(),1);
	    assert_eq!(format!("{:x?}", expect_1),
		       format!("{:x?}", got.vec));

	    let got = VmullEngine8x8::mask_start_elements(input.into(),2);
	    assert_eq!(format!("{:x?}", expect_2),
		       format!("{:x?}", got.vec));

	    let got = VmullEngine8x8::mask_start_elements(input.into(),3);
	    assert_eq!(format!("{:x?}", expect_3),
		       format!("{:x?}", got.vec));

	    let got = VmullEngine8x8::mask_start_elements(input.into(),7);
	    assert_eq!(format!("{:x?}", expect_7),
		       format!("{:x?}", got.vec));

	    let got = VmullEngine8x8::mask_start_elements(input.into(),8);
	    assert_eq!(format!("{:x?}", expect_8),
		       format!("{:x?}", got.vec));
	}
    }

}
