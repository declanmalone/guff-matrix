//! x86_64-specific SIMD
//!
//! This module contains architecture-specific code to implement the
//! "Wrap-Around Read Matrix" multiply algorithm for the x86_64
//! architecture.
//!
//! At present a fixed polynomial `0x11b` is used to implement
//! calculations in GF(2<sup>8</sup>). This may change later to
//! include a choice of different polynomials and field
//! sizes.
//!
//! 

// x86 stuff is in stable
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;



// to-do (basic vector code):
//
// * write a 64-bit version? (useful for smaller vectors/matrices)
// * write "buffer" version that works on longer (or shorter) vectors
// * test that on non-aligned data
// * test inlining of register version: are args passed on the stack?
//
// when accessing memory, I may have to implement a couple of
// different functions for arrays that may not be a full SIMD register
// in size. This happens for short arrays and final bits of longer
// arrays. The two ideas for dealing with this are:
//
// * caller guarantees that the memory off the end of the list of
//   values they want to multiply (the "guard") is readable and that
//   they don't mind it being clobbered (on the result end) (slices
//   are a natural fit with this idea, since safe Rust won't let you
//   make invalid ones).
//
// * for cases where the area following the data is either not
//   allocated or we don't want the result clobbering some other data,
//   use simd-sized buffers for staging inputs and output, and use
//   copies to/from them.
//
// I will have to look up how Rust handles simd types that are passed
// into and out of functions. I would imagine that they've got some
// sort of attribute attached to them that indicates that they can be
// passed as registers rather than going on the stack. It's a bit hard
// to find good documentation on that, though, so I might have to
// write some test code and look at the assembly output.
//
// If the SIMD types are handled specially, then I should be able to
// write some smaller pieces of code as functions. I would guess that
// even those functions with internal temporary variables (eg,
// polynomial vector in the multiply code) would qualify for constant
// elimination when used in a loop in a different function (eg,
// buffer-based multiply that repeatedly calls the SIMD multiply
// function)
//

// For matrix multiply, I'll also need to implement the infinite tape
// idea from the simulator in terms of a pair of registers for each of
// the transform, input and product stream.
//
// I want to avoid having to rewrite the same matrix multiply code for
// each of my SIMD engines, so I think that the way to approach it is
// to have two levels of macro calls:
//
// * top-level matrix code is a macro that calls things like
//   alloc_input_tape!(), advance_input_tape!() and so on.
//
// * those macros are aware of the specific SIMD types, function names
//   and so on required to implement that concept using native
//   registers, intrinsics and (my) function calls
//
// If the compiler does inlining of functions in the way I hope it
// does, then the lower-level macros can be written to use function
// calls where possible. Thus even though the macros will be in the
// matrix code module, a lot of the implementation will be delegated
// to particular SIMD modules.
//


/// 16-way SIMD multiplication of elements in GF(2<sup>8</sup>) with poly 0x11b
#[inline(always)]
pub unsafe fn vmul_p8x16(mut a : __m128i, b : __m128i, poly : u8) -> __m128i {

    let     zero = _mm_setzero_si128();
    let mut mask = _mm_set1_epi8(1);      // mask bit in b
    let     high = _mm_slli_epi32(mask,7);
    let     poly = _mm_set1_epi8(poly as i8);
    let mut res  : __m128i;
    let mut temp : __m128i;
    let mut cond : __m128i;

    // if b & mask != 0 { result = a } else { result = 0 }
    cond = _mm_cmpeq_epi8 ( _mm_and_si128(b, mask), mask);
    res  = _mm_blendv_epi8( zero, a , cond);
    // bit <<= 1
    mask = _mm_slli_epi32(mask, 1);

    // return res;

    for _ in 1..8 {
        // return res;

        // if a & 128 != 0 { a = a<<1 ^ poly } else { a = a<<1 }
        cond = _mm_cmpeq_epi8(_mm_and_si128(a, high), high);
        a    = _mm_add_epi8(a,a);
        temp = _mm_xor_si128(a , poly);
        a    = _mm_blendv_epi8(a, temp, cond);

        // return a;

        // if b & mask != 0 { result ^= a }
        cond = _mm_cmpeq_epi8 ( _mm_and_si128(b, mask), mask);
        res  = _mm_blendv_epi8( res, _mm_xor_si128(a, res), cond);
        // bit <<= 1
        mask = _mm_slli_epi32(mask, 1);
        // return res;
    }
    res
}

// multiply over buffers
//
// Important to test: non-aligned reads/writes
/// 16-way SIMD multiplication of buffers in GF(2<sup>8</sup>) with poly 0x11b
///
/// Buffers `av`, `bv` and `dest` must be a multiple of 16 in length
// #[inline(always)]
pub unsafe fn vmul_p8_buffer(dest : &mut [u8], av : &[u8], bv : &[u8], poly : u8)
{

    debug_assert_eq!(av.len(), bv.len());
    debug_assert_eq!(bv.len(), dest.len());

    let bytes = av.len();
    if bytes & 15 != 0 {
	panic!("Buffer length not a multiple of 16");
    }
    let mut times = bytes >> 4;

    // convert dest, av, bv to pointers
    let mut dest = dest.as_mut_ptr() as *mut std::arch::x86_64::__m128i;
    let mut av = av.as_ptr() as *const std::arch::x86_64::__m128i;
    let mut bv = bv.as_ptr() as *const std::arch::x86_64::__m128i;

    while times > 0 {
	times -= 1;
	let a : __m128i;
	let b : __m128i;
	let res  : __m128i;

	// read in a, b from memory
	a = _mm_lddqu_si128(av); // _mm_load_si128(av); // must be aligned
	b = _mm_lddqu_si128(bv); // b = *bv also crashes
	av = av.offset(1);	// offset for i128 type, not bytes!
	bv = bv.offset(1);

	// since the register-based multiply routine is inline, we
	// should be able to just call it here for no performancs
	// penalty. Except maybe for splatting poly each time?
	// _mm_set...

	res =  vmul_p8x16(a, b, poly);

	// unaligned version of store
	_mm_storeu_si128(dest,res);

	// *dest = res;// crashes if dest unaligned
	//return;
	dest = dest.offset(1);
    }
}





/// A function to help examine whether/how inlining is done.
/// Returns the cube of each element
pub unsafe fn vector_cube_p8x16(a : __m128i, poly : u8) -> __m128i {
    // make two calls to vmul_p8x16
    let squared = vmul_p8x16(a, a, poly);
    vmul_p8x16(squared, a, poly)
}

// Results:
//
// * no inline directive: seems to use stack (rsp, rsi?)
// * inline: still using stack
// * ...
// * inline always + native CPU: full unrolling, no stack
//
// --target-cpu=native seems to be needed to avoid _mm_blendv_epi8
// being implemented as a function call.
//
// Todo: see if passing "pointers" to simd types also avoids using the
// stack. If not, no big problem. For example, instead of:
//
// fn consume_products(&mut p0, &mut p1, words) -> consumed {}
//
// which would update the internal state registers p0, p1, removing
// the next "words" values and return another vector containing just
// the items removed ... we can use a more functional style by
// returning a tuple of (updated_p0, updated_p1, consumed)
//


// Rethink of register pair design
//
// The reason I was using two registers per matrix was that I wanted
// to be able to extract one full simd register worth of operands.
//
// That still holds.
//
// However, once I've generated the product, I don't need to store two
// registers any more.
//
// I think it's really only a matter of code clarity. The compiler
// should be able to determine that the left-hand register is no
// longer needed after the multiply, so it will be available for some
// other purpose.
//
// All three of these will happen in lock-step:
//
// * get another simd worth of xform
// * get another simd worth of input
// * multiply a simd worth of operands
//
// after that, we only need a single register to store the xform and
// input lookaheads.
//




// Alternative to macros?
//
// I had a couple of reasons for considering macros:
//
// * differing SIMD types across architectures
//
// * possibility of needing different sets of registers to accomplish
//   the same thing on different archs
//
// It's looking like the second point won't be as big a problem as I
// thought. It seems that I can write most of code in a portable way
// using only shifts and rotates as opposed to more complicated (and
// possibly more efficient) explicit masks and shuffles.


// Implement Simd-compliant structures.
//
// We will have only have a single Simd implementation here because we
// only have a single simd multiplication (cross product) routine.
//

use super::{Simd,SimdMatrix};

/// Newtype for working with __m128i vector type.
#[derive(Clone,Copy,Debug)]
pub struct X86u8x16Long0x11b {
    vec : __m128i,
}

// #![feature(const_ptr_offset)]

/// Supporting functions common to all SimdMatrix implementations here
impl X86u8x16Long0x11b {

    // left shifting pushes values further into the future
    // #[inline(always)]
    unsafe fn left_shift(reg : Self, bytes : usize) -> Self {
	// Making case of 0, 16 an error so that I can catch logic
	// errors in the calling functions.
	debug_assert!(bytes > 0);
	debug_assert!(bytes < 16);

	// eprintln!("Shifting reg left by {}", bytes);

	let no_shuffle_addr : *const u8 = SHUFFLE_MASK.as_ptr().offset(16);
	let lsh_addr = no_shuffle_addr.offset(bytes as isize * -1);
	let mask = _mm_lddqu_si128(lsh_addr as *const std::arch::x86_64::__m128i);
	Self { vec :_mm_shuffle_epi8(reg.vec, mask) }
    }

    // right shifting brings future values closer to the present
    // #[inline(always)]
    unsafe fn right_shift(reg : Self, bytes : usize) -> Self {
	// Making case of 0, 16 an error so that I can catch logic
	// errors in the calling functions.
	debug_assert!(bytes > 0);
	debug_assert!(bytes < 16);

	// eprintln!("Shifting reg right by {}", bytes);

	let no_shuffle_addr : *const u8 = SHUFFLE_MASK.as_ptr().offset(16);
	let rsh_addr = no_shuffle_addr.offset(bytes as isize);
	let mask = _mm_lddqu_si128(rsh_addr as *const std::arch::x86_64::__m128i);
	Self { vec :_mm_shuffle_epi8(reg.vec, mask) }
    }

    // #[inline(always)]
    unsafe fn combine_bytes(r0 : Self, r1: Self, bytes : usize) -> Self {

	// r0 has `bytes` values in it, zeros elsewhere
	// r1 has 16 values in it
	debug_assert!(bytes != 0);
	debug_assert!(bytes != 16);

	// eprintln!("Combining {} bytes of r0 ({:x?} with r1 ({:x?}",
	//		  bytes, r0.vec, r1.vec);
	
	// calculate r0 | (r1 >> bytes)
	Self { vec : _mm_or_si128 (r0.vec, Self::left_shift(r1, bytes).vec) }
    }

    unsafe fn future_bytes(r0 : Self, bytes : usize) -> Self {
	// shift last `bytes` values of r0 to start
	debug_assert!(bytes != 0);
	debug_assert!(bytes != 16);
	Self::right_shift(r0, 16 - bytes)
    }

}

#[allow(dead_code)]
unsafe fn test_alignr() {

    // stored in memory with lowest value first
    let av = [ 0u8, 1,  2,  3,  4,  5,  6,  7,
	        8, 9, 10, 11, 12, 13, 14, 15 ];
    let bv = [ 16u8, 17, 18, 19, 20, 21, 22, 23,
	       24, 25, 26, 27, 28, 29, 30, 31];

    let lo = _mm_lddqu_si128(av.as_ptr() as *const std::arch::x86_64::__m128i);
    let hi = _mm_lddqu_si128(bv.as_ptr() as *const std::arch::x86_64::__m128i);

    let c = _mm_alignr_epi8 (hi, lo, 31);

    eprintln!("got c = {:?}", c);

    // result of c = __m128i(31, 0) means that the low byte of memory
    // gets loaded into the low byte of the register. That means that
    // I'll have to pull off bits from the low end of a register or
    // register pair (and flip the order in a pair, too). I've
    // corrected that above.

    // In general:
    //
    // Use shr to skip lower (previously seen) memory addrs
    // Use shl to mask out later addrs that we don't want yet
    //
    // (mnemonic: the future is to the right, the past to the left)
}

/// Implement Simd trait for x86_64
///
/// Use a fixed polynomial. Even though our mul routine above can take
/// one as a parameter, my Arm vmull/vtbl implementation needs to use
/// a pre-generated lookup table. So for now, stick with a fixed poly.
impl Simd for X86u8x16Long0x11b {

    type E = u8;
    // type EE = u16;
    // type SEE = i16;
    type V = __m128i;
    const SIMD_BYTES : usize = 16;

    #[inline(always)]
    fn zero_element() -> Self::E { 0u8.into() }
    #[inline(always)]
    fn add_elements(a : Self::E, b : Self::E) -> Self::E { (a ^ b).into() }

    #[inline(always)]
    fn zero_vector() -> Self {
    	unsafe {
	    X86u8x16Long0x11b { vec :_mm_setzero_si128() }
	}
    }

    unsafe fn from_ptr(ptr: *const Self::E) -> Self {
	X86u8x16Long0x11b {
	    vec : _mm_lddqu_si128(ptr as *const std::arch::x86_64::__m128i)
	}
    }
    /// Multiply two slices, putting result in another slice.  All
    /// slices must be the same length and be a multiple of SIMD
    /// width.
    fn cross_product_slices(dest: &mut [u8],
			    av : &[u8], bv : &[u8]) {
	assert_eq!(dest.len(), av.len());
	assert_eq!(dest.len(), bv.len());
	//
	unsafe {
	    vmul_p8_buffer(&mut dest[..], &av[..], &bv[..], 0x1b);
	}
    }


    // #[inline(always)]
    fn cross_product(a : Self, b : Self) -> Self {
	unsafe {
	    Self { vec : vmul_p8x16(a.vec, b.vec, 0x1b) }
	}
    }

    // rewrite old code to use pshufb and masks

    // Two stages ...
    //
    // 1. simply use shifts to get access to required bytes
    //
    // 2. change calling program to eliminate offset requirement. We
    //    will will remove bytes from the vector(s) and return the
    //    updated vector.

    // #[inline(always)]
    unsafe fn sum_across_n(lo : Self, hi : Self, n : usize, off : usize)
			   -> (Self::E, Self) {
	// usually only called internally, so safe to use debug_assert
	debug_assert!((off < 16) && (n > 0) && (n <= 16));
	// if we straddle, will return m1 (hi), otherwise m0 (lo)
	let m = if off + n >= 16 { hi } else { lo };

	// New approach ...
	//

	// if offset != 0, shift right to skip over them
	let mut temp = lo;
	if off != 0 {
	    // eprintln!("right shifting temp (=lo) {:x?} to skip past off {}",
	    // 	      temp.vec, off);
	    temp = Self::right_shift(temp, off);
	    // eprintln!("result: {:x?}", temp);
	}

	// if are any unconsumed bytes at the end, remove them
	if off + n < 16 {
	    let shift_amount = 16 - n;
	    // eprintln!("off + n ({}) < 16", off + n);
	    // eprintln!("left shifting {:x?} by {} to remove end bytes",
	    // 	      temp.vec, shift_amount);
	    temp = Self::left_shift(temp, shift_amount);
	    // eprintln!("result: {:x?}", temp);
	}

	// note case of off + n = 16 requires neither the above
	// nor the below block.

	// if any bytes from hi are required, add them
	// (they're always at the start of the register)
	if off + n > 16 {
	    let shift_amount = 32 - (off + n);
	    // eprintln!("off + n ({}) > 16", off + n);
	    // eprintln!("left shifting hi {:x?} by {} to remove end bytes",
	    // 	      hi.vec, shift_amount);
	    let temp_hi = Self::left_shift(hi, shift_amount);
	    // eprintln!("result: {:x?}", temp_hi.vec);
	    // eprintln!("adding that to temp ({:x?})", temp.vec);
	    temp = Self { vec : _mm_xor_si128(
		temp.vec,
		temp_hi.vec
	    ) };
	    // eprintln!("result: {:x?}", temp.vec);
	}

	let mut temp = temp.vec;
	
	// sum across using fixed-size shifts
	temp = _mm_xor_si128(temp, _mm_srli_si128(temp, 8));
	temp = _mm_xor_si128(temp, _mm_srli_si128(temp, 4));
	temp = _mm_xor_si128(temp, _mm_srli_si128(temp, 2));
	temp = _mm_xor_si128(temp, _mm_srli_si128(temp, 1));
        let extracted : u8 = (_mm_extract_epi8(temp, 0) & 255) as u8;

	// eprintln!("Returning extracted {}, vector {:x?}\n", extracted, m);
	
        return (extracted, m);
    }

    // this needs more work (see Arm version for improvement)
    // #[inline(always)]
    unsafe fn read_next(mod_index : &mut usize,
			array_index : &mut usize,
			array     : &[Self::E],
			size      : usize,
			ra_size : &mut usize,
			ra : &mut Self)
			-> Self {

	let     reg0 = *ra;
	let mut reg1 : X86u8x16Long0x11b;
	let     ret  : X86u8x16Long0x11b;
	let     array_size = size;
	let     mods = *mod_index;
	let mut new_mods = mods + 16;

	// eprintln!("\n\nmods was {}, new_mods is {}", mods, new_mods);
	
	debug_assert!(mods < array_size);

	// we will always read something from array
	let addr_ptr = array.as_ptr()
	    .offset(*array_index as isize)
	    as *const std::arch::x86_64::__m128i;
	reg1 = X86u8x16Long0x11b { vec :_mm_lddqu_si128(addr_ptr) };
	*array_index += 16;

	let mut deficit = 0;
	if *array_index >= array_size && new_mods < array_size {
	    deficit = *array_index - array_size;
	}
	// eprintln!("Deficit is {}", deficit);

	let old_offset = *ra_size;

	// some bools to make logic clearer
	let will_wrap_around : bool = new_mods >= array_size;
	let had_readahead    : bool = old_offset != 0;

	if will_wrap_around {
	    // eprintln!("\n[wrapping]\n");
	    new_mods -= array_size;
	    // eprintln!("new mods {}", new_mods);

	    let want_bytes = 16 - old_offset;
	    // eprintln!("old_offset: {}", old_offset);
	    // eprintln!("want_bytes: {}", want_bytes);

	    let from_new = if want_bytes < new_mods {
		want_bytes
	    } else {
		new_mods
	    };
	    // eprintln!("from_new: {}", from_new);

	    let from_end = want_bytes - from_new;	  // from new read
	    // eprintln!("from_end: {}", from_end);

	    // reg1 <- reg0 | reg1 << old_offset
	    if old_offset == 0 {
		// noop: reg1 <- reg1
	    } else {
		// eprintln!("combining reg0, reg1");
		reg1 = X86u8x16Long0x11b
		    ::combine_bytes(reg0, reg1, old_offset);
	    }

	    // Now determine whether we need any more bytes from new
	    // stream.

	    let have_bytes = old_offset + from_end;
	    // eprintln!("have_bytes is {}", have_bytes);
	    *array_index = 0;
	    if have_bytes != 16 {

		let missing = 16 - old_offset - from_end;

		// need to read from start
		let addr_ptr = array
		    .as_ptr()
		    .offset(*array_index as isize)
		    as *const std::arch::x86_64::__m128i;
		let new = X86u8x16Long0x11b {
		    vec : _mm_lddqu_si128(addr_ptr) };
		*array_index += 16;

		// eprintln!("Taking {} bytes from new stream", missing);

		if have_bytes == 0 {
		    reg1 = new
		} else {   
		    // append part of new stream to reg1
		    // eprintln!("combining reg1 {:x?}, new {:x?}",
		    // reg1.vec, new.vec);
		    reg1 = X86u8x16Long0x11b
			::combine_bytes(reg1, new, have_bytes);
		}
		// eprintln!("new reg1 {:x?}", reg1.vec);

		// save unused part as new read-ahead
		let future_bytes = 16 - missing;
		// eprintln!("saving {} future bytes from new  {:x?}",
		// future_bytes, new.vec);
		if future_bytes != 0 {
		    *ra = X86u8x16Long0x11b::future_bytes(new, future_bytes);
		    // eprintln!("saved {:x?}", self.reg.vec);
		}

		// calculate updated ra
		*ra_size = future_bytes;

	    } else {
		*ra_size = 0
	    }

	    // save updated values and return
	    *mod_index = new_mods;
	    ret = reg1
	} else {
	
	    // This rework makes all the previously passing tests pass again.
	    if had_readahead {

		// eprintln!("combining reg0 {:x?}, reg1 {:x?}",
		// reg0.vec, reg1.vec);
		ret = X86u8x16Long0x11b::combine_bytes(reg0, reg1, old_offset);
		// eprintln!("retval {:x?}", ret.vec);

		// save unused part as new read-ahead

		let future_bytes;

		// OK. We may have got fewer than 16 bytes on the
		// read. The logic that I'm using for when wrap-around
		// happens lets that happen.
		if deficit != 0 {
		    future_bytes = old_offset - deficit;
		    *array_index = 0;
		} else {
		    future_bytes = old_offset;
		}
		*ra_size = future_bytes;
		// eprintln!("future_bytes is {}", future_bytes);
		// eprintln!("saving {} bytes from reg1  {:x?}",
		//   old_offset, reg1.vec);
		reg1 = X86u8x16Long0x11b::future_bytes(reg1, old_offset);
	    } else {
		ret = reg1;
	    }
	
	    // update state and return
	    *mod_index += 16;
	    debug_assert!(*mod_index < array_size);
	    *ra = reg1;
	}

	// eprintln!("returning {:x?}", ret.vec);
	ret
    }

}

/// Matrix storage type for x86
///
pub struct X86Matrix<S : Simd> {

    // set up a dummy value as an alternative to PhantomData
    _zero: S,

    // to implement regular matrix stuff
    rows : usize,
    cols : usize,
    pub array : Vec<u8>,
    is_rowwise : bool,
}

/// Concrete implementation of matrix for x86_64
impl X86Matrix<X86u8x16Long0x11b> {

    pub fn fill(&mut self, data : &[u8]) {
	let size = self.size();
	if data.len() != size {
	    panic!("Supplied {} data bytes  != matrix size {}",
	    data.len(), size);
	}
	self.array[0..size].copy_from_slice(data);
    }

    pub fn new_with_data(rows : usize, cols : usize, is_rowwise : bool,
		     data : &[u8]) -> Self {
	let mut this = Self::new(rows, cols, is_rowwise);
	this.fill(data);
	this
    }

}

// table for pshufb (_mm_shuffle_epi8())
// Entries with high bit set => 0
// Other entries => select byte at that index
const SHUFFLE_MASK : [u8; 48] = [
    255u8, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
    0,   1,   2,   3,   4,   5,   6,   7,
    8,   9,  10,  11,  12,  13,  14,  15,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255,
];

use guff::F8;

impl SimdMatrix<X86u8x16Long0x11b,F8> for X86Matrix<X86u8x16Long0x11b> {

    fn new(rows : usize, cols : usize, is_rowwise : bool) -> Self {
	let size = rows * cols;
	if size < 16 {
	 //   panic!("This matrix can't handle rows * cols < 16 bytes");
	}

	// add an extra 15 guard bytes beyond size
	let array = vec![0u8; size + 15];

	// set up a dummy value as an alternative to PhantomData
	let _zero = X86u8x16Long0x11b::zero_vector();
	
	X86Matrix::<X86u8x16Long0x11b> {
	    rows, cols, is_rowwise, array, _zero
	}
    }

    #[inline(always)]
    fn rows(&self) -> usize { self.rows }

    #[inline(always)]
    fn cols(&self) -> usize { self.cols }

    #[inline(always)]
    fn is_rowwise(&self) -> bool { self.is_rowwise }

    fn as_slice(&self) -> &[u8] {
	let size = self.size();
	&self.array[0..size]
    }

    #[inline(always)]
    fn indexed_read(&self, index : usize) -> u8 {
	self.array[index]
    }

    #[inline(always)]
    fn indexed_write(&mut self, index : usize, elem : u8) {
	self.array[index] = elem;
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
	let size = self.size();
	&mut self.array[0..size]
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    // if we ever need to use a reference multiply:
    // use guff::{GaloisField,new_gf8};

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    #[allow(unused_imports)]
    use std::arch::x86_64::*;

    #[test]
    fn test_vmul_p8x16() {
	unsafe  {
	    #[allow(overflowing_literals)]
	    let b : __m128i = _mm_set_epi32( 0xca000202, 0x000053ca, 0x00000000, 0x00000102 );
	    #[allow(overflowing_literals)]
            let a : __m128i = _mm_set_epi32( 0x53004080, 0x0000ca53, 0x00000000, 0x00000201 );
	    let c : __m128i = _mm_set_epi32( 0x0100801b, 0x00000101, 0x00000000, 0x00000202 );
	    let result = vmul_p8x16(a, b, 0x1b);
	    
	    // eprintln!("a     : {:?}", a );
	    // eprintln!("b     : {:?}", b );
	    // eprintln!("result: {:?}", result   );
	    // eprintln!("expect: {:?}", c );
	    // panic!();
	    assert_eq!(format!("{:?}", c), format!("{:?}", result))
	}
    }

    #[test]
    fn test_vmul_p8_buffer() {
	unsafe  {

	    // can only use on struct, enum, fn or union ...
	    // #[repr(align(16))]
	    let a = [0x53u8; 160];
	    let b = [0xcau8; 160];
	    let mut d = [0x00u8; 160];
	    let i = [0x01u8; 160]; // expect identity {53} x {ca}

	    vmul_p8_buffer(&mut d[..], &a, &b, 0x1b);
	    
	    // eprintln!("a     : {:?}", a );
	    // eprintln!("b     : {:?}", b );
	    // eprintln!("result: {:?}", result   );
	    // eprintln!("expect: {:?}", c );
	    // panic!();
	    assert_eq!(format!("{:?}", d), format!("{:?}", i))
	}
    }

    // crashes as expected if I use _mm_load_si128(av);
    #[test]
    fn test_vmul_p8_buffer_unaligned_read() {
	unsafe  {

	    // can only use on struct, enum, fn or union ...
	    // #[repr(align(16))]
	    let a = [0x53u8; 161];
	    let b = [0xcau8; 161];
	    let mut d = [0x00u8; 160];
	    let i = [0x01u8; 160]; // expect identity {53} x {ca}

	    vmul_p8_buffer(&mut d[..], &a[1..], &b[1..], 0x1b);
	    
	    // eprintln!("a     : {:?}", a );
	    // eprintln!("b     : {:?}", b );
	    // eprintln!("result: {:?}", d   );
	    // eprintln!("expect: {:?}", i );
	    // panic!();
	    assert_eq!(format!("{:?}", d), format!("{:?}", i))
	}
    }

    // crashes as expected if I use _mm_store_si128() or * to deref
    #[test]
    fn test_vmul_p8_buffer_unaligned_write() {
	unsafe  {

	    // can only use on struct, enum, fn or union ...
	    // #[repr(align(16))]
	    let a = [0x53u8; 160];
	    let b = [0xcau8; 160];
	    let mut d = [0u8; 161];
	    let i = [0x01u8; 160]; // expect identity {53} x {ca}

	    vmul_p8_buffer(&mut d[1..], &a[..], &b[..], 0x1b);
	    
	    // eprintln!("a     : {:?}", a );
	    // eprintln!("b     : {:?}", b );
	    // eprintln!("result: {:?}", d   );
	    // eprintln!("expect: {:?}", i );
	    // panic!();
	    assert_eq!(format!("{:?}", &d[1..161]), format!("{:?}", &i[0..160]))
	}
    }

    #[test]
    fn test_alignr_shr() {
	unsafe { test_alignr() };
    }

    #[test]
    fn test_sum_across_n() {
	// first byte of av is stored in lowest memory location
	let av = [ 0u8, 1,  2,  4,  8, 16, 32, 64,
	           128, 0,  1,  2,  4,  8, 16, 32, ];
	let bv = [ 1u8, 2,  4,  8, 16, 32, 64, 128,
		     0, 1,  2,  4,  8, 16, 32, 64,];

	unsafe {

	    // av[0] goes into low byte of lo
	    let lo = _mm_lddqu_si128(av.as_ptr() as *const std::arch::x86_64::__m128i);
	    let hi = _mm_lddqu_si128(bv.as_ptr() as *const std::arch::x86_64::__m128i);

	    // wrap the registers up in Simd type
	    let lo = X86u8x16Long0x11b { vec : lo };
	    let hi = X86u8x16Long0x11b { vec : hi };

	    // simplest case 
	    let (sum,_new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 16, 0);
	    let expect : u8 = 0b0111_1111 ^ 0b1011_1111;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);

	    // n = power of two 
	    let (sum,_new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 8, 0);
	    assert_eq!(sum, 0b0111_1111);

	    // simplest case, with offset 1
	    let (sum,_new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 16, 1);
	    let expect : u8 = 0b1111_1111 ^ 0b0011_1110;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);

	    // off = 0, n = 1
	    let (sum,_new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 1, 0);
	    assert_eq!(sum, 0b0000_0000);
	    
	    // off = 0, n = 2
	    let (sum,_new_m)
		= X86u8x16Long0x11b::sum_across_n(lo, hi, 2, 0);
	    assert_eq!(sum, 0b0000_0001);

	    // off = 0, n = 3
	    let (sum,_new_m)
		= X86u8x16Long0x11b::sum_across_n(lo, hi, 3, 0);
	    assert_eq!(sum, 0b0000_0011);

	    // off = 0, n = 4
	    let (sum,_new_m)
		= X86u8x16Long0x11b::sum_across_n(lo, hi, 4, 0);
	    assert_eq!(sum, 0b0000_0111);

	    // off = 0, n = 5
	    let (sum,_new_m)
		= X86u8x16Long0x11b::sum_across_n(lo, hi, 5, 0);
	    assert_eq!(sum, 0b0000_1111);

	    // off = 0, n = 6
	    let (sum,_new_m)
		= X86u8x16Long0x11b::sum_across_n(lo, hi, 6, 0);
	    assert_eq!(sum, 0b0001_1111);

	    // off = 0, n = 7
	    let (sum,_new_m)
		= X86u8x16Long0x11b::sum_across_n(lo, hi, 7, 0);
	    assert_eq!(sum, 0b0011_1111);

	    // off = 0, n = 15
	    let (sum,_new_m)
		= X86u8x16Long0x11b::sum_across_n(lo, hi, 15, 0);
	    let expect : u8 = 0b0111_1111 ^ 0b1001_1111;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);
	}
    }

    #[test]
    fn test_new_sum_across_n() {
	// first byte of av is stored in lowest memory location
	let av = [ 0u8, 1,  2,  4,  8, 16, 32, 64,
	           128, 0,  1,  2,  4,  8, 16, 32, ];
	let bv = [ 1u8, 2,  4,  8, 16, 32, 64, 128,
		     0, 1,  2,  4,  8, 16, 32, 64,];

	unsafe {

	    // av[0] goes into low byte of lo
	    let lo = _mm_lddqu_si128(
		av.as_ptr() as *const std::arch::x86_64::__m128i);
	    let hi = _mm_lddqu_si128(
		bv.as_ptr() as *const std::arch::x86_64::__m128i);

	    // wrap the registers up in Simd type
	    let lo = X86u8x16Long0x11b { vec : lo };
	    let hi = X86u8x16Long0x11b { vec : hi };

	    // try different offsets
	    
	    let (sum,_new_m)
		= X86u8x16Long0x11b::sum_across_n(lo, hi, 16, 3);
	    let expect : u8 = 0b1111_1101 ^ 0b0011_1001;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);

	    let (sum,_new_m)
		= X86u8x16Long0x11b::sum_across_n(lo, hi, 1, 3);
	    let expect : u8 = 4;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);

	    let (sum,_new_m)
		= X86u8x16Long0x11b::sum_across_n(lo, hi, 2, 3);
	    let expect : u8 = 4 + 8;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);

	}
    }

    
    // test constructor basics first, then iterator
    #[test]
    #[should_panic]
    fn test_matrix_too_small() {
	let _ = X86Matrix::<X86u8x16Long0x11b>::new(3, 5, true);
    }

    #[test]
    fn test_matrix_goldilocks() {
	let _ = X86Matrix::<X86u8x16Long0x11b>::new(2, 8, true);
	let _ = X86Matrix::<X86u8x16Long0x11b>::new(8, 2, true);
	let _ = X86Matrix::<X86u8x16Long0x11b>::new(16, 1, true);
	let _ = X86Matrix::<X86u8x16Long0x11b>::new(4, 4, true);
    }

    #[test]
    fn test_matrix_read_pre_fill() {

	let     mat = X86Matrix::<X86u8x16Long0x11b>::new(4, 4, true);

	let mut mat_mod_index = 0;
	let mut mat_array_index = 0;
	let     mat_array = mat.as_slice();
	let     mat_size  = mat.size();
	let mut mat_ra_size = 0;
	let mut mat_ra = X86u8x16Long0x11b::zero_vector();
	
	unsafe {
        let zero : __m128i = _mm_set_epi32( 0, 0, 0, 0 );
	    let first_read = X86u8x16Long0x11b::read_next(
		&mut mat_mod_index,
		&mut mat_array_index,
		mat_array,
		mat_size,
		&mut mat_ra_size,
		&mut mat_ra);
	    assert_eq!(format!("{:?}",zero), format!("{:?}",first_read.vec))
	}
    }

    #[test]
    fn test_matrix_read_post_fill() {

	let mut mat = X86Matrix::<X86u8x16Long0x11b>::new(4, 4, true);

	let identity = [ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 ];
	mat.fill(&identity[..]);

	let mut mat_mod_index = 0;
	let mut mat_array_index = 0;
	let     mat_array = mat.as_slice();
	let     mat_size  = mat.size();
	let mut mat_ra_size = 0;
	let mut mat_ra = X86u8x16Long0x11b::zero_vector();

	unsafe {

	    // first operand = lowest 32 bits
            let one : __m128i = _mm_set_epi32(
		0x01000000,	// why big endian, though?
		0x00010000,
		0x00000100,
		0x00000001
	    );

	    // just to be sure that the above is correct:
	    let array_ptr = identity
		.as_ptr() as *const std::arch::x86_64::__m128i;
	    let id_reg = _mm_lddqu_si128(array_ptr);
	    assert_eq!(format!("{:?}",one), format!("{:?}",id_reg));

	    // now test read_next()
	    let first_read = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);
	    assert_eq!(format!("{:?}",one),
		       format!("{:?}",first_read.vec));

	    // That big-end ordering for u32 load is puzzling, so make
	    // sure that when we write register back to memory it's in
	    // correct order

	    let mut scratch = [0u8; 16];
	    let scratch_ptr = scratch
		.as_mut_ptr() as *mut std::arch::x86_64::__m128i;

	    _mm_storeu_si128(scratch_ptr,first_read.vec);
	    assert_eq!(scratch, identity);
	}
    }

    #[test]
    fn test_matrix_easy_wraparound() {

	// simplest case where matrix size == simd size
	let mut mat = X86Matrix::<X86u8x16Long0x11b>::new(4, 4, true);

	let identity = [ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 ];
	mat.fill(&identity[..]);
	let mut mat_mod_index = 0;
	let mut mat_array_index = 0;
	let     mat_array = mat.as_slice();
	let     mat_size  = mat.size();
	let mut mat_ra_size = 0;
	let mut mat_ra = X86u8x16Long0x11b::zero_vector();
	

	unsafe {

	    // first operand = lowest 32 bits
            let one : __m128i = _mm_set_epi32(
		0x01000000,	// why big endian, though?
		0x00010000,
		0x00000100,
		0x00000001
	    );

	    // just to be sure that the above is correct:
	    let array_ptr = identity.as_ptr() as *const std::arch::x86_64::__m128i;
	    let id_reg = _mm_lddqu_si128(array_ptr);
	    assert_eq!(format!("{:x?}",one), format!("{:x?}",id_reg));

	    // first read_next() already tested above
	    let read = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);
	    assert_eq!(format!("{:x?}",one), format!("{:x?}",read.vec));

	    // test second read_next(), should equal first
	    let read = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);
	    assert_eq!(format!("{:x?}",one), format!("{:x?}",read.vec));

	    // test third read_next(), should equal first
	    let read = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);
	    assert_eq!(format!("{:x?}",one), format!("{:x?}",read.vec));
	}
    }

    #[test]
    fn test_matrix_internal_read() { // 3 non-wrapping + 2 wrapping

	// case where matrix size is a multiple of simd size
	let mut mat = X86Matrix::<X86u8x16Long0x11b>::new(16, 3, true);

	// use constant shuffle table which has 16 * 3 elements
	mat.fill(&SHUFFLE_MASK[..]);
	let mut mat_mod_index = 0;
	let mut mat_array_index = 0;
	let     mat_array = mat.as_slice();
	let     mat_size  = mat.size();
	let mut mat_ra_size = 0;
	let mut mat_ra = X86u8x16Long0x11b::zero_vector();
	

	unsafe {

	    // load up each 16-byte "row" of SHUFFLE_MASK
	    // one row of 0xff, one row of incrementing values, then another 0xff
	    
	    let array_ptr = SHUFFLE_MASK.as_ptr();
	    let ff1_addr = array_ptr.offset( 0) as *const std::arch::x86_64::__m128i;
	    let inc_addr = array_ptr.offset(16) as *const std::arch::x86_64::__m128i;
	    let ff2_addr = array_ptr.offset(32) as *const std::arch::x86_64::__m128i;
	    
	    let ff1 =  _mm_lddqu_si128(ff1_addr);
	    let inc =  _mm_lddqu_si128(inc_addr);
	    let ff2 =  _mm_lddqu_si128(ff2_addr);

	    // just to be sure that the above is correct:
	    assert_eq!(format!("{:x?}",ff1), format!("{:x?}",ff2));

	    // 1st
	    let read = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);
	    assert_eq!(format!("{:x?}",ff1), format!("{:x?}",read.vec));

	    // 2nd
	    let read = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);
	    assert_eq!(format!("{:x?}",inc), format!("{:x?}",read.vec));

	    // 3rd
	    let read = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);
	    assert_eq!(format!("{:x?}",ff1), format!("{:x?}",read.vec));

	    // 4th
	    let read = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);
	    assert_eq!(format!("{:x?}",ff1), format!("{:x?}",read.vec));

	    // 5th
	    let read = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);
	    assert_eq!(format!("{:x?}",inc), format!("{:x?}",read.vec));
	}
    }

    #[test]
    fn test_matrix_changing_read_offset() {

	// case where matrix size is not a multiple of simd size

	// A 7x3 matrix would generate all possible offsets since
	// gcd(21,16) = 1

	// 2x matrix storage so we can easily generate overlapping
	// reads using a modular index
	let stream = [0u8,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
		      0u8,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
	
	let mut mat = X86Matrix::<X86u8x16Long0x11b>::new(7, 3, true);

	mat.fill(&stream[0..21]);

	let mut mat_mod_index = 0;
	let mut mat_array_index = 0;
	let     mat_array = mat.as_slice();
	let     mat_size  = mat.size();
	let mut mat_ra_size = 0;
	let mut mat_ra = X86u8x16Long0x11b::zero_vector();
	

	let array_ptr = stream.as_ptr();
	let mut index = 0;

	unsafe {

	    // do this in a loop that tests all pathways in code
	    // lcm(21,16) = 21 * 16. Do *2 +1 to check proper restart

	    for _ in 0..21*16 * 2 + 1 {
	    
		// load up expected value
		let addr = array_ptr.offset(index)
		    as *const std::arch::x86_64::__m128i;
		let expect =  _mm_lddqu_si128(addr);

		index += 16;
		if index >= 21 { index -= 21 }

		let mat_read = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);

		assert_eq!(format!("{:x?}",mat_read.vec),
			   format!("{:x?}",expect));
	    }
	}
    }

    // More thorough testing of read_next with a variety of matrix sizes
    #[test]
    fn test_read_next_cycles() {

	let mut errors = 0;
	for rows in 4..18 {
	    for cols in 4..23 {

		// no restrictions on how many cols, since we're only
		// calling read_next, not doing matrix multiply
		let size = rows * cols;
		let mut mat = X86Matrix::<X86u8x16Long0x11b>
		    ::new(rows, cols, true);

		let fill_list = (1u8..=255).cycle().take(size);
		let fill_vec : Vec<u8> = fill_list.collect();

		eprintln!("filling matrix with {} bytes", fill_vec.len());
		mat.fill(&fill_vec[..]);

		let mut mat_mod_index = 0;
		let mut mat_array_index = 0;
		let     mat_array = mat.as_slice();
		let     mat_size  = mat.size();
		let mut mat_ra_size = 0;
		let mut mat_ra = X86u8x16Long0x11b::zero_vector();
		
		let mut ref_list = (1u8..=255).cycle().take(size).cycle();
		let mut ref_vec = [0u8; 16];

		for i in 0 .. size {
		    unsafe {
			let from_mat = X86u8x16Long0x11b::read_next(
			    &mut mat_mod_index,
			    &mut mat_array_index,
			    mat_array,
			    mat_size,
			    &mut mat_ra_size,
			    &mut mat_ra);

			for i in 0..16 {
			    ref_vec[i] = ref_list.next().unwrap();
			}
			let addr = ref_vec.as_ptr()
			    as *const std::arch::x86_64::__m128i;
			let expect =  _mm_lddqu_si128(addr);

			let fmt_ref = format!("{:x?}",expect);
			let fmt_mat = format!("{:x?}",from_mat.vec);

			if fmt_mat != fmt_ref {
			    eprintln!("read_next() failed");
			    eprintln!("Matrix {} rows x {} columns ",
				      rows, cols);
			    eprintln!("Got {} != ref {} at position {}",
				      fmt_mat, fmt_ref, i);
			    errors += 1;
			}
		    }
		}
	    }
	}
	if errors > 0 {
	    panic!("Failing test: {} errors", errors);
	}
    }
}
