//! x86_64-specific SIMD

// x86 stuff is in stable
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;


// to-do (basic vector code):
//
// * write a 64-bit version? (useful for smaller vectors)
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


// multiply
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

// A function to help examine whether/how inlining is done.
// Returns the cube of each element
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

}
