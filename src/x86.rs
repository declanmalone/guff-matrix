//! x86_64-specific SIMD

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
#[inline(always)]
pub unsafe fn vmul_p8_buffer(dest : &mut [u8], av : &[u8], bv : &[u8], poly : u8) {

    assert_eq!(av.len(), bv.len());
    assert_eq!(bv.len(), dest.len());

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
	b = _mm_lddqu_si128(bv); // *bv also crashes
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

use super::Simd;

#[derive(Clone,Copy,Debug)]
struct X86u8x16Long0x11b {
    vec : __m128i,
}

// Add extra things here to help test intrinsics
//impl X86u8x16Long0x11b {

  //  fn 
//}

unsafe fn test_alignr() {

    // stored in memory with lowest value first
    let av = [ 0u8, 1,  2,  3,  4,  5,  6,  7,
	        8, 9, 10, 11, 12, 13, 14, 15 ];
    let bv = [ 16u8, 17, 18, 19, 20, 21, 22, 23,
	       24, 25, 26, 27, 28, 29, 30, 31];
    
    let mut lo = _mm_lddqu_si128(av.as_ptr() as *const std::arch::x86_64::__m128i);
    let mut hi = _mm_lddqu_si128(bv.as_ptr() as *const std::arch::x86_64::__m128i);

    let c = _mm_alignr_epi8 (hi, lo, 31);

    panic!("got c = {:?}", c);

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

// Use fixed polynomial. Even though our mul routine above can take
// one as a parameter, my Arm vmull/vtbl implementation needs to use a
// pre-generated lookup table. So for now, stick with a fixed poly.
impl Simd for X86u8x16Long0x11b {

    type E = u8;
    type V = __m128i;
    
    fn cross_product(a : Self, b : Self) -> Self {
	unsafe {
	    Self { vec : vmul_p8x16(a.vec, b.vec, 0x1b) }
	}
    }
    // renaming variable lo: current, hi: future readahead
    fn sum_across_n(lo : Self, hi : Self, mut n : usize, off : usize)
		    -> (Self::E, Self) {
	unsafe {
	// now the fun(?) starts ... looking for intrinsics
	// to implement this ...

	// __m128i _mm_alignr_epi8 (__m128i a, __m128i b, int imm8)
	// sticks a, b together (a high), then shifts right by imm8
	//
	// This will work fine if the leftmost bytes have already been
	// zeroed.
	//
	// What is the mapping of memory addresses to high, low bytes
	// of __m128i, though? Will we have to reverse the direction
	// of shifts? Intel is little-endian
	//
	// Right. See test_alignr above. The order is the reverse to
	// what I had expected. I can still use it combined with:
	//
	// __m128i _mm_slli_si128 (__m128i a, int imm8)
	// (left shift imm8 bytes)
	//

	// if we straddle, will return m1 (hi), otherwise m0 (lo)
	let m = if off + n >= 16 { hi } else { lo };

	// Abandoning the code below 'OK: alignr...'. It seems that
	// Intel don't provide full vector rotates except for constant
	// values. I'm going to have to use shuffle masks...

	// Looking at the Arm intrinsics, it doesn't have it either. I
	// guess that the circuitry required for doing arbitrary
	// rotates on wide registers is too costly.

	// So, I guess that both architectures do something like
	// Altivec does...
	//
	// pshufb on Intel seems to work by:
	//
	// __m128i _mm_shuffle_epi8 (__m128i a, __m128i b)
	//
	// IF high bit of mask b is set, zero corresponding output
	// element
	// ELSE
	// use lower 4 bits of b to select a byte from a
	//
	// My Altivec/PS3 code actually uses different scheme ... it
	// uses shuffles to advance past already-consumed data (as
	// above), but then uses maskb to set a number of bits from
	// the start. (converting a 16-bit value into a 16-byte
	// vector).	

	// There's probably something similar for Intel?
	// 

	// This has turned out to be more complex than I thought. I
	// might have to rethink the matrix multiply code. The point
	// at which control passes here might have to be at a lower
	// level, meaning that we do less work in the matrix code and
	// more here.

	// Actually, my sum across products is also in doubt. Ah, no,
	// it's fine. We can still shift by a constant amount.

	// That gives me an idea... the following can be converted
	// into binary searches. Hopefully, though, it compiles down
	// to a computed goto (preferably adding a constant multiple
	// to pc)
	let mut c;
	match off {
	    0 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 0) },
	    1 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 1) },
	    2 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 2) },
	    3 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 3) },
	    4 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 4) },
	    5 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 5) },
	    6 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 6) },
	    7 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 7) },
	    8 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 8) },
	    9 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 9) },
	    10 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 10) },
	    11 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 11) },
	    12 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 12) },
	    13 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 13) },
	    14 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 14) },
	    15 => { c = _mm_alignr_epi8 (hi.vec, lo.vec, 15) },
	    _ => { c = hi.vec } 	// unreachable, but satisfy compiler
	}

	// That only gets rid of the first `off` bytes. We also then
	// have to select the next n bytes to sum together.

	// if this looks crazy, it's because it is
	if n == 16 {
	    n >>= 1
	} else {
	    match n {
		15 => { c = _mm_slli_si128(_mm_srli_si128(c, 1), 1) },
		14 => { c = _mm_slli_si128(_mm_srli_si128(c, 2), 2) },
		13 => { c = _mm_slli_si128(_mm_srli_si128(c, 3), 3) },
		12 => { c = _mm_slli_si128(_mm_srli_si128(c, 4), 4) },
		11 => { c = _mm_slli_si128(_mm_srli_si128(c, 5), 5) },
		10 => { c = _mm_slli_si128(_mm_srli_si128(c, 6), 6) },
		 9 => { c = _mm_slli_si128(_mm_srli_si128(c, 7), 7) },
		_ => {}
	    }
	    n &= 7
	}
	c = _mm_xor_si128(c, _mm_slli_si128(c, 8));
	if n == 8 {
	    n >>= 1
	} else {
	    match n {
		7 => { c = _mm_slli_si128(_mm_srli_si128(c, 9), 9) },
		6 => { c = _mm_slli_si128(_mm_srli_si128(c, 10), 10) },
		5 => { c = _mm_slli_si128(_mm_srli_si128(c, 11), 11) },
		_ => {}
	    }
	    n &= 3
	}
	c = _mm_xor_si128(c, _mm_slli_si128(c, 4));
	if n == 4 {
	    n >>= 1
	} else {
	    match n {
		3 => { c = _mm_slli_si128(_mm_srli_si128(c, 13), 13) },
		_ => {}
	    }
	    n &= 1
	}
	c = _mm_xor_si128(c, _mm_slli_si128(c, 2));
	if n == 2 {
	    c = _mm_xor_si128(c, _mm_slli_si128(c, 1));
	}
        return ((_mm_extract_epi8(c, 0) & 256) as u8, m);
	
	// OK: alignr doesn't work because the offset has to be a
	// constant. Plan B.

	// extract from off ... off + n
	// let mut c = _mm_alignr_epi8 (hi.vec, lo.vec, 0);

	// OMG: slli also has to use a const
	// let lshift = 16 - n;
	// c = _mm_slli_si128(c, lshift);

	// sum across using xor and shift/rotate
	//
	// 16 bytes, so 4 steps
	// c ^= _mm_slli_si128(c, 8);
	
	// (0, m)
    }
}
}
// We can have several different matrix implementations, each with
// their own way of implementing read_next(). For example, we could
// have variants such as:
//
// * caching complete matrix in a small number of registers
// * 




#[cfg(test)]
mod tests {

    use super::*;
    use guff::{GaloisField,new_gf8};

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
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
}
