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

#[derive(Clone,Copy,Debug)]
pub struct X86u8x16Long0x11b {
    vec : __m128i,
}

// #![feature(const_ptr_offset)]

// Add extra things here to help test intrinsics
impl X86u8x16Long0x11b {

    // left shifting pushes values further into the future
    #[inline(always)]
    unsafe fn left_shift(reg : Self, bytes : usize) -> Self {
	// Making case of 0, 16 an error so that I can catch logic
	// errors in the calling functions.
	assert!(bytes > 0);
	assert!(bytes < 16);

	// eprintln!("Shifting reg left by {}", bytes);

	let NO_SHUFFLE_ADDR : *const u8 = SHUFFLE_MASK.as_ptr().offset(16);
	let lsh_addr = NO_SHUFFLE_ADDR.offset(bytes as isize * -1);
	let mask = _mm_lddqu_si128(lsh_addr as *const std::arch::x86_64::__m128i);
	Self { vec :_mm_shuffle_epi8(reg.vec, mask) }
    }

    // right shifting brings future values closer to the present
    #[inline(always)]
    unsafe fn right_shift(reg : Self, bytes : usize) -> Self {
	// Making case of 0, 16 an error so that I can catch logic
	// errors in the calling functions.
	assert!(bytes > 0);
	assert!(bytes < 16);

	// eprintln!("Shifting reg right by {}", bytes);

	let NO_SHUFFLE_ADDR : *const u8 = SHUFFLE_MASK.as_ptr().offset(16);
	let rsh_addr = NO_SHUFFLE_ADDR.offset(bytes as isize);
	let mask = _mm_lddqu_si128(rsh_addr as *const std::arch::x86_64::__m128i);
	Self { vec :_mm_shuffle_epi8(reg.vec, mask) }
    }

    #[inline(always)]
    unsafe fn combine_bytes(r0 : Self, r1: Self, bytes : usize) -> Self {

	// r0 has `bytes` values in it, zeros elsewhere
	// r1 has 16 values in it
	assert!(bytes != 0);
	assert!(bytes != 16);

	// eprintln!("Combining {} bytes of r0 ({:x?} with r1 ({:x?}",
	//		  bytes, r0.vec, r1.vec);
	
	// calculate r0 | (r1 >> bytes)
	Self { vec : _mm_or_si128 (r0.vec, Self::left_shift(r1, bytes).vec) }
    }

    unsafe fn future_bytes(r0 : Self, bytes : usize) -> Self {
	// shift last `bytes` values of r0 to start
	assert!(bytes != 0);
	assert!(bytes != 16);
	Self::right_shift(r0, 16 - bytes)
    }
	
}

unsafe fn test_alignr() {

    // stored in memory with lowest value first
    let av = [ 0u8, 1,  2,  3,  4,  5,  6,  7,
	        8, 9, 10, 11, 12, 13, 14, 15 ];
    let bv = [ 16u8, 17, 18, 19, 20, 21, 22, 23,
	       24, 25, 26, 27, 28, 29, 30, 31];
    
    let mut lo = _mm_lddqu_si128(av.as_ptr() as *const std::arch::x86_64::__m128i);
    let mut hi = _mm_lddqu_si128(bv.as_ptr() as *const std::arch::x86_64::__m128i);

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

// Use fixed polynomial. Even though our mul routine above can take
// one as a parameter, my Arm vmull/vtbl implementation needs to use a
// pre-generated lookup table. So for now, stick with a fixed poly.
impl Simd for X86u8x16Long0x11b {

    type E = u8;
    type V = __m128i;
    const SIMD_BYTES : usize = 16;

    #[inline(always)]
    fn zero_element() -> Self::E { 0u8.into() }
    #[inline(always)]
    fn add_elements(a : Self::E, b : Self::E) -> Self::E { (a ^ b).into() }
    
    #[inline(always)]
    fn cross_product(a : Self, b : Self) -> Self {
	unsafe {
	    Self { vec : vmul_p8x16(a.vec, b.vec, 0x1b) }
	}
    }
    // renaming variable lo: current, hi: future readahead
    #[target_feature(enable = "ssse3")]
    //#[target_feature(enable = "4.1")]
    //#[target_feature(enable = "sse41")]
    unsafe fn sum_across_n_old(lo : Self, hi : Self, mut n : usize, off : usize)
			   -> (Self::E, Self) {
	// TODO: rewrite this using pshufb and masks
	assert!((off < 16) && (n > 0) && (n <= 16));

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

	// eprintln!("Taking {} bytes starting at offset {}", n, off);
	// eprintln!("lo vector: {:x?}", lo);
	// eprintln!("hi vector: {:x?}", hi);
	
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
	    _ => { c = hi.vec },	// unreachable, but satisfy compiler
	}

	// eprintln!("c after alignr: {:x?}", c); 
	
	// That only gets rid of the first `off` bytes. We also then
	// have to select the next n bytes to sum together.

	// Each pair of shl, shr below removes some number of high
	// bytes from the register (shl), then moves everything back
	// to the original position (shr)
	//
	// We repeatedly:
	// * clear some number of high bytes from readahead
	// * xor the top half of the remaining bytes into the bottom
	// * halve the number of bytes
	//
	// Eventually, we should end up with the correct sum in
	// the low byte.

	// Actually, the shr is not needed? If we accumulate in
	// the high byte, perhaps?
	//
	
	// Xor across n byte elements of a 128-bit simd register:
	// * Masks out bytes above n using shifts,
	// * xors by half remaining step size
	// * log_2 (max_n) steps
        match n {
            16 => {},
            15 => { c = _mm_srli_si128(_mm_slli_si128(c, 1), 1) },
            14 => { c = _mm_srli_si128(_mm_slli_si128(c, 2), 2) },
            13 => { c = _mm_srli_si128(_mm_slli_si128(c, 3), 3) },
            12 => { c = _mm_srli_si128(_mm_slli_si128(c, 4), 4) },
            11 => { c = _mm_srli_si128(_mm_slli_si128(c, 5), 5) },
            10 => { c = _mm_srli_si128(_mm_slli_si128(c, 6), 6) },
            9  => { c = _mm_srli_si128(_mm_slli_si128(c, 7), 7) },
            _  => { c = _mm_srli_si128(_mm_slli_si128(c, 8), 8) },
        }
        if n > 8 { n = 8 }
        c = _mm_xor_si128(c, _mm_srli_si128(c, 8));
        // eprintln!("c after first xor: {:x?}", c);

        // eprintln!("n is now {}", n);
        match n {
            8 => { },
            7 => { c = _mm_srli_si128(_mm_slli_si128(c, 9), 9) },
            6 => { c = _mm_srli_si128(_mm_slli_si128(c, 10), 10) },
            5 => { c = _mm_srli_si128(_mm_slli_si128(c, 11), 11) },
            _ => { c = _mm_srli_si128(_mm_slli_si128(c, 12), 12) },
        }
        if n > 4 { n = 4 }
        c = _mm_xor_si128(c, _mm_srli_si128(c, 4));
        // eprintln!("c after second xor: {:x?}", c);

        // eprintln!("n is now {}", n);
        match n {
            4 => { },
            3 => { c = _mm_srli_si128(_mm_slli_si128(c, 13), 13) },
            _ => { c = _mm_srli_si128(_mm_slli_si128(c, 14), 14) },
        }
        if n > 2 { n = 2 }
        c = _mm_xor_si128(c, _mm_srli_si128(c, 2));
        // eprintln!("c after third xor: {:x?}", c);

        // eprintln!("n is now {}", n);
        match n {
            2 => { },
            _ => { c = _mm_slli_si128(_mm_srli_si128(c, 15), 15) },
        }
        c = _mm_xor_si128(c, _mm_srli_si128(c, 1));
        // eprintln!("c after fourth xor: {:x?}", c);
        let extracted : u8 = (_mm_extract_epi8(c, 0) & 255) as u8;
        // eprintln!("Extracting low byte: {:x}", extracted);
        return (extracted, m);
        
    }

    // rewrite old code to use pshufb and masks

    // Two stages ...
    //
    // 1. simply use shifts to get access to required bytes
    //
    // 2. change calling program to eliminate offset requirement. We
    //    will will remove bytes from the vector(s) and return the
    //    updated vector.

    #[inline(always)]
    unsafe fn sum_across_n(lo : Self, hi : Self, mut n : usize, off : usize)
			   -> (Self::E, Self) {
	assert!((off < 16) && (n > 0) && (n <= 16));
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
}


// We can have several different matrix implementations, each with
// their own way of implementing read_next(). For example, we could
// have variants such as:
//
// * caching complete matrix in a small number of registers
// * work over memory, but for small matrix sizes (needs extra checks)
// * work over memory with "safe" (not small) sizes
// 

// First version ... the latter ... matrices are at least as big as
// simd. In fact, make it greater than twice the simd size. Smallest
// usable identity matrix is then 4x4 = 16.
//
// My reason for choosing this is that it makes read-ahead easier if
// we can start off (in the constructor) by reading in a full register
// without needing to worry about wrap-around.

pub struct X86SimpleMatrix<S : Simd> {

    // to implement read_next
    reg    : S,			// single read-ahead register
    ra     : usize,		// readahead amount (how full is reg)
    rp     : usize,		// read pointer (index) in array
    mods   : usize,		// counter mod size of array
    
    // to implement write_next
    or : usize,
    oc : usize,

    // to implement regular matrix stuff
    rows : usize,
    cols : usize,
    pub array : Vec<u8>,
    is_rowwise : bool,
}

impl X86SimpleMatrix<X86u8x16Long0x11b> {

    pub fn new(rows : usize, cols : usize, is_rowwise : bool) -> Self {
	if rows * cols < 16 {
	    panic!("This SIMD matrix implementation can't handle rows * cols < 16 bytes");
	}

	let or = 0;
	let oc = 0;
	let ra = 0;
	let rp = 0;
	let mods = 0;

	// add an extra 15 guard bytes to deal with reading past end
	// of matrix
	let array = vec![0u8; rows * cols + 15];

	// set up a dummy value for reg; set it up properly after fill()
	let reg;
	unsafe {
	    reg = X86u8x16Long0x11b { vec :_mm_setzero_si128() };
	}

	X86SimpleMatrix::<X86u8x16Long0x11b> {
	    rows, cols, is_rowwise, array,
	    reg, rp, ra, or, oc, mods
	}
    }

    pub fn fill(&mut self, data : &[u8]) {
	let size = self.size();
	if data.len() != size {
	    panic!("Supplied {} data bytes  != matrix size {}",
	    data.len(), size);
	}
	self.array[0..size].copy_from_slice(data);

	unsafe {
	    // Set up read-ahead register based on new data
	    // let array_ptr = self.array.as_ptr() as *const std::arch::x86_64::__m128i;

	    // risk an aligned load (use repr align on struct if it fails)
	    // self.reg = X86u8x16Long0x11b{vec : _mm_load_si128(array_ptr)};
	    
	    // read-ahead pointer = simd width, since we have read one full register
	    self.reg = X86u8x16Long0x11b { vec :_mm_setzero_si128() };
	}

	self.ra  = 0; 		// read-ahead amount
	self.rp = 0;
	self.mods = 0;
    }

    // convenience
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
    255u8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
    255u8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

//
// Doing all aligned reads, eg simd width 4, matrix size 5
//
// Reads at A and E are aligned:
//
// Matrix     ABCD EABC DEAB CDEA BCDE
//            ABCD
//                 E---
//                  ABC D
//                       E-- -
//                        AB CD
//                             E- --
//                              A BCD
//                                   E ---
//
//            ABCD EABC DEAB CDEA BCDE
//
// DEAB calculation shows full complexity...
//
// * have D left over from previous step
// * reads off end of matrix
// * has to start off reading again from A
// * combines DEAB to give current SIMD answer
// * stashes CD for next step
//
// The next one, CDEA is similar.
//
// Correct code hinges on modular arithmetic, particularly the value
// of `position % matrix size`
//
// ABCD calculation:
//
// pos (0,4): no wrap
// ra was 0, so full ABCD read returned
//
// EABC calculation:
//
// pos  (4,8) overflows giving new pos 8-5 = 3
// ra was 0, so prev + E = E (null shuffle)
// take 4-3 = 1 from end, 3 from new giving EABC
// ra becomes 1, containing D
//
// DEAB calculation:
//
// pos (3,7) overflows giving new pos 7-5 = 2
// ra was 1, so add D+E
// take 4-2 = 2 from DE and 2 from new giving DEAB
// ra becomes 4 - 2 = 2, containing CD
//
// CDEA calculation:
//
// pos (2,6) overflows giving new pos 6-5 = 1
// ra was 2, so add CD + E giving CDE
// take 4-1 = 3 from CDE and 1 from new, giving CDEA
// ra becomes 4 - 1 = 3, containing BCD
//
// BCDE calculation:
//
// pos (1,5): overflows giving new pos 5-5 = 0
// ra was 3, so add BCD + E
// take 4 - 0 = 4 from BCDE and 0 from new, giving BCDE
// ra becomes 4 - 0 = 4 containing ABCD
//
// ABCD calculation (restart, but this time with readahead of 4!)
//
// pos (0,4): no wrap
// ra was 4, so add ABCD + null
// take 4 - 4 = 0 from ABCD and 4 from new giving E---
//
// So, the last calculation isn't correct. It can be avoided by not
// reading from the new stream in the final BCDE calculation of the
// first cycle.
//
// ie, if we would ever combine 0 bytes from the new stream, we won't
// read them in this step. So, ra will always be calculated mod 4.
//
// This makes sense because we then have two interlocking counters,
// one mod 5 and one mod 4, and the cycle will continue back at the
// start again after lcm(4,5) = byte 20 reads.

impl SimdMatrix<X86u8x16Long0x11b> for X86SimpleMatrix<X86u8x16Long0x11b> {

    const SIMD_SIZE : usize = 128;

    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn is_rowwise(&self) -> bool { self.is_rowwise }

    unsafe fn read_next(&mut self) -> X86u8x16Long0x11b {
	// Stub. Interestingly, this would be sufficient for a 4x4 matrix
	// return self.reg;

	let     reg0 = self.reg;
	let mut reg1 : X86u8x16Long0x11b;
	let mut reg2 : X86u8x16Long0x11b;
	let     ret  : X86u8x16Long0x11b;
	let     array_size = self.rows * self.cols;
	let     mods = self.mods;
	let mut new_mods = mods + 16;

	// another rework needed. 
	//
	// move all wrap-around to top
	
	// eprintln!("\n\nmods was {}, new_mods is {}", mods, new_mods);
	
	assert!(mods < array_size);

	// we will always read something from array
	let addr_ptr = self.array.as_ptr().offset(self.rp as isize) as *const std::arch::x86_64::__m128i;
	reg1 = X86u8x16Long0x11b { vec :_mm_lddqu_si128(addr_ptr) };
	self.rp += 16;

	let mut deficit = 0;
	if self.rp >= array_size && new_mods < array_size {
	    deficit = self.rp - array_size;
	}
	// eprintln!("Deficit is {}", deficit);
	
	let old_offset = self.ra;
	let missing = 16 - old_offset;

	// some bools to make logic clearer
	let will_wrap_around : bool = (new_mods + 15 >= array_size + 15) || (new_mods >= array_size);
	let had_readahead    : bool = old_offset != 0;
	let will_read_again  : bool = will_wrap_around && (old_offset + missing != 16);

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
	    }; // from reg1
	    // eprintln!("from_new: {}", from_new);

	    let from_end = want_bytes - from_new;	  // from new read
	    // eprintln!("from_end: {}", from_end);

	    
	    // reg1 <- reg0 | reg1 << old_offset
	    if old_offset == 0 {
		// noop: reg1 <- reg1
	    } else {
		// eprintln!("combining reg0, reg1");
		reg1 = X86u8x16Long0x11b::combine_bytes(reg0, reg1, old_offset);
	    }

	    // Now determine whether we need any more bytes from new
	    // stream.

	    let have_bytes = old_offset + from_end;
	    // eprintln!("have_bytes is {}", have_bytes);
	    self.rp = 0;
	    if have_bytes != 16 {

		let missing = 16 - old_offset - from_end;

		// need to read from start
		let addr_ptr = self.array
		    .as_ptr()
		    .offset(self.rp as isize) as *const std::arch::x86_64::__m128i;
		let new = X86u8x16Long0x11b { vec : _mm_lddqu_si128(addr_ptr) };
		self.rp += 16;

		// eprintln!("Will take {} bytes from new stream",  missing);

		if have_bytes == 0 {
		    reg1 = new
		} else {   
		    // append part of new stream to reg1
		    // eprintln!("combining reg1 {:x?}, new {:x?}", reg1.vec, new.vec);
		    reg1 = X86u8x16Long0x11b::combine_bytes(reg1, new, have_bytes);
		}
		// eprintln!("new reg1 {:x?}", reg1.vec);

		// save unused part as new read-ahead
		let future_bytes = 16 - missing;
		// eprintln!("saving {} future bytes from new  {:x?}", future_bytes, new.vec);
		if future_bytes != 0 {
		    self.reg = X86u8x16Long0x11b::future_bytes(new, future_bytes);
		    // eprintln!("saved {:x?}", self.reg.vec);
		}

		// calculate updated ra
		self.ra = future_bytes;

	    } else {
		
		self.ra = 0
	    }

	    // save updated values and return
	    self.mods = new_mods;

	    // return value
	    ret = reg1
	} else {
	
	    // This rework makes all the previously passing tests pass again.

	    // <= because there's no straddling and we can continue rp at zero again
	    // if mods + 16 <= array_size {

	    // can safely read without wrap-around
	    // eprintln!("\n[not wrapping]\n");
	    // eprintln!("old_offset: {}", old_offset);

	    let missing = 16 - old_offset;

	    // if we have no partial reads from before, must merge
	    // that with this and save new remainder

	    if old_offset != 0 {

		// eprintln!("combining reg0 {:x?}, reg1 {:x?}", reg0.vec, reg1.vec);
		ret = X86u8x16Long0x11b::combine_bytes(reg0, reg1, old_offset);
		// eprintln!("retval {:x?}", ret.vec);

		// save unused part as new read-ahead

		let future_bytes;

		// OK. We may have got fewer than 16 bytes on the
		// read. The logic that I'm using for when wrap-around
		// happens lets that happen.
		if deficit != 0 {
		    future_bytes = old_offset - deficit;
		    self.rp = 0;
		} else {
		    future_bytes = old_offset;
		}
		self.ra = future_bytes;
		// eprintln!("future_bytes is {}", future_bytes);
		// eprintln!("saving {} bytes from reg1  {:x?}", old_offset, reg1.vec);
		reg1 = X86u8x16Long0x11b::future_bytes(reg1, old_offset); // saved later
	    } else {
		ret = reg1;
	    }
	
	    // update state and return
	    //self.rp   += 16;
	    self.mods += 16;
	    if self.mods >= array_size {
		panic!();
		// self.mods -= array_size;
		// self.rp = 0;
	    }
	    self.reg = reg1;
	}

	// eprintln!("returning {:x?}", ret.vec);
	ret
    }
    fn write_next(&mut self, e : u8) {

	let or = self.or;
	let oc = self.oc;

	let index = self.rowcol_to_index(or, oc);
	self.array[index] = e;

	self.or = if or + 1 < self.rows { or + 1 } else { 0 };
	self.oc = if oc + 1 < self.cols { oc + 1 } else { 0 };

    }
}



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

    #[test]
    fn test_sum_across_n() {
	// first byte of av is stored in lowest memory location
	let av = [ 0u8, 1,  2,  4,  8, 16, 32, 64,
	           128, 0,  1,  2,  4,  8, 16, 32, ];
	let bv = [ 1u8, 2,  4,  8, 16, 32, 64, 128,
		     0, 1,  2,  4,  8, 16, 32, 64,];

	unsafe {

	    // av[0] goes into low byte of lo
	    let mut lo = _mm_lddqu_si128(av.as_ptr() as *const std::arch::x86_64::__m128i);
	    let mut hi = _mm_lddqu_si128(bv.as_ptr() as *const std::arch::x86_64::__m128i);

	    // wrap the registers up in Simd type
	    let mut lo = X86u8x16Long0x11b { vec : lo };
	    let mut hi = X86u8x16Long0x11b { vec : hi };

	    // simplest case 
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 16, 0);
	    let expect : u8 = 0b0111_1111 ^ 0b1011_1111;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);

	    // n = power of two 
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 8, 0);
	    assert_eq!(sum, 0b0111_1111);

	    // simplest case, with offset 1
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 16, 1);
	    let expect : u8 = 0b1111_1111 ^ 0b0011_1110;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);

	    // off = 0, n = 1
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 1, 0);
	    assert_eq!(sum, 0b0000_0000);
	    
	    // off = 0, n = 2
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 2, 0);
	    assert_eq!(sum, 0b0000_0001);

	    // off = 0, n = 3
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 3, 0);
	    assert_eq!(sum, 0b0000_0011);

	    // off = 0, n = 4
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 4, 0);
	    assert_eq!(sum, 0b0000_0111);

	    // off = 0, n = 5
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 5, 0);
	    assert_eq!(sum, 0b0000_1111);

	    // off = 0, n = 6
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 6, 0);
	    assert_eq!(sum, 0b0001_1111);

	    // off = 0, n = 7
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 7, 0);
	    assert_eq!(sum, 0b0011_1111);

	    // off = 0, n = 15
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 15, 0);
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
	    let mut lo = _mm_lddqu_si128(av.as_ptr() as *const std::arch::x86_64::__m128i);
	    let mut hi = _mm_lddqu_si128(bv.as_ptr() as *const std::arch::x86_64::__m128i);

	    // wrap the registers up in Simd type
	    let mut lo = X86u8x16Long0x11b { vec : lo };
	    let mut hi = X86u8x16Long0x11b { vec : hi };

	    // try different offsets, etc.
	    
	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 16, 3);
	    let expect : u8 = 0b1111_1101 ^ 0b0011_1001;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);

	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 1, 3);
	    let expect : u8 = 4;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);

	    let (sum,new_m) = X86u8x16Long0x11b::sum_across_n(lo, hi, 2, 3);
	    let expect : u8 = 4 + 8;
	    eprintln!("expect {:x}", expect);
	    assert_eq!(sum, expect);

	}
    }

    
    // test constructor basics first, then iterator
    #[test]
    #[should_panic]
    fn test_matrix_too_small() {
	let mat = X86SimpleMatrix::<X86u8x16Long0x11b>::new(3, 5, true);
    }

    #[test]
    fn test_matrix_goldilocks() {
	let mat = X86SimpleMatrix::<X86u8x16Long0x11b>::new(2, 8, true);
	let mat = X86SimpleMatrix::<X86u8x16Long0x11b>::new(8, 2, true);
	let mat = X86SimpleMatrix::<X86u8x16Long0x11b>::new(16, 1, true);
	let mat = X86SimpleMatrix::<X86u8x16Long0x11b>::new(4, 4, true);
    }

    #[test]
    fn test_matrix_read_pre_fill() {
	let mut mat = X86SimpleMatrix::<X86u8x16Long0x11b>::new(4, 4, true);
	unsafe {
        let zero : __m128i = _mm_set_epi32( 0, 0, 0, 0 );
	    let first_read = mat.read_next();
	    assert_eq!(format!("{:?}",zero), format!("{:?}",first_read.vec))
	}
    }

    #[test]
    fn test_matrix_read_post_fill() {

	let mut mat = X86SimpleMatrix::<X86u8x16Long0x11b>::new(4, 4, true);

	let identity = [ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 ];
	mat.fill(&identity[..]);

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
	    assert_eq!(format!("{:?}",one), format!("{:?}",id_reg));

	    // now test read_next()
	    let first_read = mat.read_next();
	    assert_eq!(format!("{:?}",one), format!("{:?}",first_read.vec));

	    // That big-end ordering for u32 load is puzzling, so make
	    // sure that when we write register back to memory it's in
	    // correct order

	    let mut scratch = [0u8; 16];
	    let scratch_ptr = scratch.as_mut_ptr() as *mut std::arch::x86_64::__m128i;

	    _mm_storeu_si128(scratch_ptr,first_read.vec);
	    assert_eq!(scratch, identity);
	}
    }

    #[test]
    fn test_matrix_easy_wraparound() {

	// simplest case where matrix size == simd size
	let mut mat = X86SimpleMatrix::<X86u8x16Long0x11b>::new(4, 4, true);

	let identity = [ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 ];
	mat.fill(&identity[..]);

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
	    let read = mat.read_next();
	    assert_eq!(format!("{:x?}",one), format!("{:x?}",read.vec));

	    // test second read_next(), should equal first
	    let read = mat.read_next();
	    assert_eq!(format!("{:x?}",one), format!("{:x?}",read.vec));

	    // test third read_next(), should equal first
	    let read = mat.read_next();
	    assert_eq!(format!("{:x?}",one), format!("{:x?}",read.vec));
	}
    }

    #[test]
    fn test_matrix_internal_read() { // 3 non-wrapping + 2 wrapping

	// case where matrix size is a multiple of simd size
	let mut mat = X86SimpleMatrix::<X86u8x16Long0x11b>::new(16, 3, true);

	// use constant shuffle table which has 16 * 3 elements
	mat.fill(&SHUFFLE_MASK[..]);

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
	    let read = mat.read_next();
	    assert_eq!(format!("{:x?}",ff1), format!("{:x?}",read.vec));

	    // 2nd
	    let read = mat.read_next();
	    assert_eq!(format!("{:x?}",inc), format!("{:x?}",read.vec));

	    // 3rd
	    let read = mat.read_next();
	    assert_eq!(format!("{:x?}",ff1), format!("{:x?}",read.vec));

	    // 4th
	    let read = mat.read_next();
	    assert_eq!(format!("{:x?}",ff1), format!("{:x?}",read.vec));

	    // 5th
	    let read = mat.read_next();
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
	
	let mut mat = X86SimpleMatrix::<X86u8x16Long0x11b>::new(7, 3, true);

	mat.fill(&stream[0..21]);

	let array_ptr = stream.as_ptr();
	let mut index = 0;

	unsafe {

	    // do this in a loop that tests all pathways in code
	    // lcm(21,16) = 21 * 16. Do +1 to check proper restart

	    for _ in 0..21*16 + 1 {
	    
		// load up expected value
		let addr = array_ptr.offset(index) as *const std::arch::x86_64::__m128i;
		let expect =  _mm_lddqu_si128(addr);

		index += 16;
		if index >= 21 { index -= 21 }

		let mat_read = mat.read_next();

		assert_eq!(format!("{:x?}",mat_read.vec), format!("{:x?}",expect));
	    }
	}
    }


}
