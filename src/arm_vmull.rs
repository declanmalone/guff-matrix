//#![feature(stdsimd)]


#[cfg(target_arch = "arm")]
use core::arch::arm::*;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use std::mem::transmute;

use crate::*;

// Arm-specific matrix multiply with new dot product apportionment
// code based on ..
///
// rotating masks for selection...
//
// use tbl1
//
// interested in extracting up to k (or k % simd) elements
//
// start with a mask of 0..k-1 (selects k from start of vector)
//
// all other elements are 0x80 or some sufficiently-large out-of-range
// value. BUT: can't be too high because we will be adding and
// subtracting values from the mask, and selecting, eg, 0xff for
// "blanked" elements will end up selecting them at some point since
// they'll end up in the range 0..7
//
// evolution of mask given, eg, k=3:
//
// 0,1,2   full read of 3 bytes
// 3,4,5   full read of next 3 bytes
// 6,7,8   read 2 bytes from this 8-byte vector
//
// This straddles the boundary. The "8" value is outside the valid
// range of lookup values, so we only read (apportion) bytes 6 and 7.
//
// Each of the above mask updates simply add 3 to the mask values.
//
// When we straddle, we subtract the simd width, 8, giving:
//
// -2,-1,0
//
// This sets us up to read the 1 single byte that should be
// apportioned to this dot product (albeit in the third byte of the
// vector, but that doesn't matter).
//
// A similar scheme works for the case where k > simd width (where we
// have periodic additions of k % simd followed by subtractions of
// simd)
//
// (the case where k is congruent to the simd width can be treated
// specially, since there is no need to do apportionment of partial
// sums)
//
// This is useful because:
//
// * we only have to consider the current simd vector (not m0,m1)
// * vtbl1 has no extra latency under A64
// * we get to reuse the same (single) mask during apportionment
// * operations are only addition and subtraction of value to mask
//   (splat and add)
//
// I'm only considering this for Arm/vtbl right now, but something
// similar (not as good) involving left and right shifts should also
// work.

// Separate out different cases of how k relates to simd
//
// k < simd
//
// In this case we start off with a value from this table:
//
const SELECT_K_FROM_START : [u8; 8 * 9] =
    [
        128, 128, 128, 128,    128, 128, 128, 128, // select none
        0  , 128, 128, 128,    128, 128, 128, 128, // select 1
        0  ,   1, 128, 128,    128, 128, 128, 128,
        0  ,   1,   2, 128,    128, 128, 128, 128,
        0  ,   1,   2,   3,    128, 128, 128, 128,
        0  ,   1,   2,   3,      4, 128, 128, 128,
        0  ,   1,   2,   3,      4,   5, 128, 128,
        0  ,   1,   2,   3,      4,   5,   6, 128,
        0  ,   1,   2,   3,      4,   5,   6,   7, // select all
    ];
//
// eg, if k = 3, we use the 0,1,2,128,... line
//
// After each read, we add k to all elements in the mask
//
// When we read past the end of the list of products, we subtract 8
// from all elements in the mask.

// Case where simd divides k (in other words, k is a multiple of simd;
// k = 8,16,24, etc.)
//
// No problem with dot products straddling simd boundary at all


// Case where k > simd

// for k > simd, it's probably better to use a different scheme. The
// above (k < simd) scheme will work fine for selecting bytes at the
// start of the vector, but then we'll want to apportion all the
// remaining bytes to the next dot product.
//
// This seems to require using two masks...

// Won't use this though! See below.
const SELECT_K_FROM_END : [u8; 8 * 9] =
    [
        128, 128, 128, 128,    128, 128, 128, 128, // select none
        128, 128, 128, 128,    128, 128, 128,   7, // select 1
        128, 128, 128, 128,    128, 128,   6,   7, // 
        128, 128, 128, 128,    128,   5,   6,   7, // 
        128, 128, 128, 128,      4,   5,   6,   7, // 
        128, 128, 128,   3,      4,   5,   6,   7, // 
        128, 128,   2,   3,      4,   5,   6,   7, // 
        128,   1,   2,   3,      4,   5,   6,   7, // 
        0  ,   1,   2,   3,      4,   5,   6,   7, // select all
    ];

// with simd > k, can we update those masks to represent the shifting
// boundary between this dp and the next? Yes, if we store what are
// effectively shift tables:

// code for shift right
const APPORTION_FROM_HIGHER : [u8; 8 * 8] =
    [
        8 , 9, 10,  11,     12,   13,   14,   15 , // select none
        7 , 8,  9,  10,     11,   12,   13,   14 , // select 1
        6 , 7,  8,   9,     10,   11,   12,   13 , 
        5 , 6,  7,   8,      9,   10,   11,   12 , 
        4 , 5,  6,   7,      8,    9,   10,   11 , 
        3 , 4,  5,   6,      7,    8,    9,   10 , 
        2 , 3,  4,   5,      6,    7,    8,   9  , 
        1 , 2,  3,   4,      5,    6,    7,   8  ,

        // remove repeated/shared entry:
        // 0 , 1,  2,   3,      4,    5,    6,   7  , // select all
    ];

// code for shift left
const APPORTION_FROM_LOWER : [i8; 8 * 9] =
    [
        0  ,   1,   2,   3,      4,   5,   6,   7, // select all
        -1 ,   0,   1,   2,      3,   4,   5,   6,
        -2 ,  -1,   0,   1,      2,   3,   4,   5,
        -3 ,  -2,  -1,   0,      1,   2,   3,   4,
        -4 ,  -3,  -2,  -1,      0,   1,   2,   3,
        -5 ,  -4,  -3,  -2,     -1,   0,   1,   2,
        -6 ,  -5,  -4,  -3,     -2,  -1,   0,   1,
        -7 ,  -6,  -5,  -4,     -3,  -2,  -1,   0, // select 1
        -8 ,  -7,  -6,  -5,     -4,  -3,  -2,  -1, // select none
    ];

// Note that I've written down the entire table, but values need not
// be re-read from the table. Also, the two tables are continuations
// of each other.
//
// To use these tables ...
//
// set up left/right shift masks that are 8 apart in the combined
// table above (eg, one is select none, the other select all, or
// select 1, select 7).
//
// if we're not straddling the boundary, apportion all 8 bytes to the
// current sum.
//
// The straddling point moves k to the right each time we cross a
// boundary. We add k to each element in each mask. If this would move
// us out of range, subtract 8.
//
// It seems that an alternative, using actual bitmasks of 0x00 and
// 0xff would also work, but we'll still end up using the same shift
// mechanics as above, so we won't really gain much benefit, if any.
//



pub fn apportion_mask(k : isize) -> VmullEngine8x8 {
    debug_assert!(k < 8);
    unsafe {
        let addr = SELECT_K_FROM_START.as_ptr();
        VmullEngine8x8::read_simd(addr.offset(k * 8))
    }
}

pub fn update_apportion_mask(mask : VmullEngine8x8, _k : isize)
                             -> VmullEngine8x8 {
    unsafe {
        mask
    }
}

// switcher/despatcher based on k <=> simd width
//
// later on, I will probably add more special cases where the
// transform matrix can fit into 1--4 simd registers
pub fn arm_simd_matrix_mul<S : Simd<E=G::E> + Copy, G>(
    xform  : &mut impl SimdMatrix<S,G>,
    input  : &mut impl SimdMatrix<S,G>,
    output : &mut impl SimdMatrix<S,G>)
where S::E : Copy + Zero + One, G : GaloisField
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
    
    if n > 1 {
        let denominator = gcd(n,c);
        debug_assert_ne!(n, denominator);
        debug_assert_ne!(c, denominator);
    }

    if k & 7 == 0 {
        unsafe {
            arm_matrix_mul_k_multiple_simd(xform, input, output)
        }
    } else {
        if k > 8 {
            panic!();
            unsafe {
                arm_matrix_mul_k_gt_simd(xform, input, output)
            }
        } else {
            panic!();
            unsafe {
                arm_matrix_mul_k_lt_simd(xform, input, output)
            }
        }
    }
}

// special case where k is a multiple of simd
pub unsafe fn arm_matrix_mul_k_multiple_simd<S : Simd<E=G::E> + Copy, G>(
    xform  : &mut impl SimdMatrix<S,G>,
    input  : &mut impl SimdMatrix<S,G>,
    output : &mut impl SimdMatrix<S,G>)
where S::E : Copy + Zero + One, G : GaloisField {

    let c = input.cols();
    let n = xform.rows();
    let k = xform.cols();

    // isn't a pub fn, so debug_assert is OK
    debug_assert!(k & 7 == 0);
    
    // algorithm not so trivial any more, but still quite simple
    let mut dp_counter  = 0;
    let mut sum         = S::zero_element();
    let zero = S::zero_vector();
    let mut sum_vector  = zero;

    let simd_width = 8;

    // Code for read_next_with_mask() that was handled in SimdMatrix has now
    // moved to Simd. We need to track those variables here.
    let mut xform_mod_index = 0;
    let mut xform_array_index = 0;
    let     xform_array = xform.as_slice();
    let     xform_size  = xform.size();
    let mut xform_ra_size = 0;
    let mut xform_ra = zero;
    let mut xform_mask = S::starting_mask();

    let mut input_mod_index = 0;
    let mut input_array_index = 0;
    let     input_array = input.as_slice();
    let     input_size  = input.size();
    let mut input_ra_size = 0;
    let mut input_ra = zero;
    let mut input_mask = S::starting_mask();

    // we handle or and oc (was in matrix class)
    let mut or : usize = 0;
    let mut oc : usize = 0;
    let orows = output.rows();
    let ocols = output.cols();

    let mut i0 : S;
    let mut x0 : S;
    let mut m0 : S;

    let mut total_dps = 0;
    let target = n * c;         // number of dot products

    // Be agnostic about layout
    let right = if output.is_rowwise() { 1 } else { n };
    let down  = if output.is_rowwise() { c } else { 1 };

    while total_dps < target {

        while dp_counter < k {

            // TODO: avail of the fact that there are no straddling
            // reads, so no need to track readahead

            let addr = xform_array.as_ptr()
                .offset(xform_mod_index) as *const u8;
            let read_ptr = xform_array.as_ptr()
                .offset((xform_array_index) as isize);
            x0 = VmullEngine8x8::read_simd(read_ptr as *const u8).into();
            xform_array_index += 8;
            if xform_array_index == xform_size {
                xform_array_index = 0
            }
            
            // x0 = S::read_next_with_mask(&mut xform_mod_index,
            //                   &mut xform_array_index,
            //                   xform_array,
            //                   xform_size,
            //                   &mut xform_ra_size,
            //                   &mut xform_ra,
            //                   &mut xform_mask);
            i0 = S::read_next_with_mask(&mut input_mod_index,
                              &mut input_array_index,
                              input_array,
                              input_size,
                              &mut input_ra_size,
                              &mut input_ra,
                              &mut input_mask);

            m0  = S::cross_product(x0,i0);

            sum_vector = S::sum_vectors(&sum_vector,&m0);
            // sum ^= S::sum_across_simd(m0);
            dp_counter += simd_width;
        }

        // sum now has a full dot product
        // eprintln!("Sum: {}", sum);

        let sum = S::sum_across_simd(sum_vector);

        // handle writing and incrementing or, oc
        let write_index = or * down + oc * right;
        output.indexed_write(write_index,sum);
        or = if or + 1 < orows { or + 1 } else { 0 };
        oc = if oc + 1 < ocols { oc + 1 } else { 0 };

        sum_vector = zero;
        dp_counter = 0;
        total_dps += 1;
    }
}

unsafe fn arm_matrix_mul_k_gt_simd<S : Simd<E=G::E> + Copy, G>(
    xform  : &mut impl SimdMatrix<S,G>,
    input  : &mut impl SimdMatrix<S,G>,
    output : &mut impl SimdMatrix<S,G>)
where S::E : Copy + Zero + One, G : GaloisField {

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
    // let mut sum         = S::zero_element();
    let zero = S::zero_vector();
    let mut sum_vector  = zero;
    
    
    let simd_width = S::SIMD_BYTES;

    // Code for read_next_with_mask() that was handled in SimdMatrix has now
    // moved to Simd. We need to track those variables here.
    let mut xform_mod_index = 0;
    let mut xform_array_index = 0;
    let     xform_array = xform.as_slice();
    let     xform_size  = xform.size();
    let mut xform_ra_size = 0;
    let mut xform_ra = zero;
    let mut xform_mask = S::starting_mask();

    let mut input_mod_index = 0;
    let mut input_array_index = 0;
    let     input_array = input.as_slice();
    let     input_size  = input.size();
    let mut input_ra_size = 0;
    let mut input_ra = zero;
    let mut input_mask = S::starting_mask();

    // we handle or and oc (was in matrix class)
    let mut or : usize = 0;
    let mut oc : usize = 0;
    let orows = output.rows();
    let ocols = output.cols();

    // read ahead two products

    let mut i0 : S;
    let mut x0 : S;

    x0 = S::read_next_with_mask(&mut xform_mod_index,
                      &mut xform_array_index,
                      xform_array,
                      xform_size,
                      &mut xform_ra_size,
                      &mut xform_ra,
                      &mut xform_mask);
    i0 = S::read_next_with_mask(&mut input_mod_index,
                      &mut input_array_index,
                      input_array,
                      input_size,
                      &mut input_ra_size,
                      &mut input_ra,
                      &mut input_mask);

    let mut m0 = S::cross_product(x0,i0);

    x0 = S::read_next_with_mask(&mut xform_mod_index,
                      &mut xform_array_index,
                      xform_array,
                      xform_size,
                      &mut xform_ra_size,
                      &mut xform_ra,
                      &mut xform_mask);
    i0 = S::read_next_with_mask(&mut input_mod_index,
                      &mut input_array_index,
                      input_array,
                      input_size,
                      &mut input_ra_size,
                      &mut input_ra,
                      &mut input_mask);
    let mut m1  = S::cross_product(x0,i0);

    let mut offset_mod_simd = 0;
    let mut total_dps = 0;
    let target = n * c;         // number of dot products

    // Be agnostic about layout
    let right = if output.is_rowwise() { 1 } else { n };
    let down  = if output.is_rowwise() { c } else { 1 };

    while total_dps < target {

        // at top of loop we should always have m0, m1 full

        // apportion parts of m0,m1 to sum

        // handle case where k >= simd_width
        while dp_counter + simd_width <= k {
            // let (part, new_m)
            // = S::sum_across_n(m0,m1,simd_width,offset_mod_simd);

            let (part, new_m)                                       
                = S::extract_elements(m0,m1,simd_width,offset_mod_simd);
            sum_vector = S::sum_vectors(&sum_vector,&part);

            m0 = new_m;
            // x0  = xform.read_next_with_mask();
            // i0  = input.read_next_with_mask();
            x0 = S::read_next_with_mask(&mut xform_mod_index,
                              &mut xform_array_index,
                              xform_array,
                              xform_size,
                              &mut xform_ra_size,
                              &mut xform_ra,
                              &mut xform_mask);
            i0 = S::read_next_with_mask(&mut input_mod_index,
                              &mut input_array_index,
                              input_array,
                              input_size,
                              &mut input_ra_size,
                              &mut input_ra,
                              &mut input_mask);
            m1  = S::cross_product(x0,i0); // new m1
            dp_counter += simd_width;
            // offset_mod_simd unchanged
        }
        // above may have set dp_counter to k already.
        if dp_counter < k {            // If not, ...
            let want = k - dp_counter; // always strictly positive

            // eprintln!("Calling sum_across_n with m0 {:?}, m1 {:?}, n {}, offset {}",
            //      m0.vec, m1.vec, want, offset_mod_simd);
            // let (part, new_m) = S::sum_across_n(m0,m1,want,offset_mod_simd);

            let (part, new_m) = S::extract_elements(m0,m1,want,offset_mod_simd);
            
            // eprintln!("got sum {}, new m {:?}", part, new_m.vec);

            sum_vector = S::sum_vectors(&sum_vector,&part);
            if offset_mod_simd + want >= simd_width {
                // consumed m0 and maybe some of m1 too
                m0 = new_m;     // nothing left in old m0, so m0 <- m1
                // x0  = xform.read_next_with_mask();
                // i0  = input.read_next_with_mask();
                x0 = S::read_next_with_mask(&mut xform_mod_index,
                                  &mut xform_array_index,
                                  xform_array,
                                  xform_size,
                                  &mut xform_ra_size,
                                  &mut xform_ra,
                                  &mut xform_mask);
                i0 = S::read_next_with_mask(&mut input_mod_index,
                                  &mut input_array_index,
                                  input_array,
                                  input_size,
                                  &mut input_ra_size,
                                  &mut input_ra,
                                  &mut input_mask);
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

        let sum = S::sum_across_simd(sum_vector);

        // handle writing and incrementing or, oc
        let write_index = or * down + oc * right;
        output.indexed_write(write_index,sum);
        or = if or + 1 < orows { or + 1 } else { 0 };
        oc = if oc + 1 < ocols { oc + 1 } else { 0 };

        sum_vector = zero;
        dp_counter = 0;
        total_dps += 1;
    }
}

unsafe fn arm_matrix_mul_k_lt_simd<S : Simd<E=G::E> + Copy, G>(
    xform  : &mut impl SimdMatrix<S,G>,
    input  : &mut impl SimdMatrix<S,G>,
    output : &mut impl SimdMatrix<S,G>)
where S::E : Copy + Zero + One, G : GaloisField {

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
    // let mut sum         = S::zero_element();
    let zero = S::zero_vector();
    let mut sum_vector  = zero;
    
    
    let simd_width = S::SIMD_BYTES;

    // Code for read_next_with_mask() that was handled in SimdMatrix has now
    // moved to Simd. We need to track those variables here.
    let mut xform_mod_index = 0;
    let mut xform_array_index = 0;
    let     xform_array = xform.as_slice();
    let     xform_size  = xform.size();
    let mut xform_ra_size = 0;
    let mut xform_ra = zero;
    let mut xform_mask = S::starting_mask();

    let mut input_mod_index = 0;
    let mut input_array_index = 0;
    let     input_array = input.as_slice();
    let     input_size  = input.size();
    let mut input_ra_size = 0;
    let mut input_ra = zero;
    let mut input_mask = S::starting_mask();

    // we handle or and oc (was in matrix class)
    let mut or : usize = 0;
    let mut oc : usize = 0;
    let orows = output.rows();
    let ocols = output.cols();

    // read ahead two products

    let mut i0 : S;
    let mut x0 : S;

    x0 = S::read_next_with_mask(&mut xform_mod_index,
                      &mut xform_array_index,
                      xform_array,
                      xform_size,
                      &mut xform_ra_size,
                      &mut xform_ra,
                      &mut xform_mask);
    i0 = S::read_next_with_mask(&mut input_mod_index,
                      &mut input_array_index,
                      input_array,
                      input_size,
                      &mut input_ra_size,
                      &mut input_ra,
                      &mut input_mask);

    let mut m0 = S::cross_product(x0,i0);

    x0 = S::read_next_with_mask(&mut xform_mod_index,
                      &mut xform_array_index,
                      xform_array,
                      xform_size,
                      &mut xform_ra_size,
                      &mut xform_ra,
                      &mut xform_mask);
    i0 = S::read_next_with_mask(&mut input_mod_index,
                      &mut input_array_index,
                      input_array,
                      input_size,
                      &mut input_ra_size,
                      &mut input_ra,
                      &mut input_mask);
    let mut m1  = S::cross_product(x0,i0);

    let mut offset_mod_simd = 0;
    let mut total_dps = 0;
    let target = n * c;         // number of dot products

    // Be agnostic about layout
    let right = if output.is_rowwise() { 1 } else { n };
    let down  = if output.is_rowwise() { c } else { 1 };

    while total_dps < target {

        // at top of loop we should always have m0, m1 full

        // apportion parts of m0,m1 to sum

        // handle case where k >= simd_width
        while dp_counter + simd_width <= k {
            // let (part, new_m)
            // = S::sum_across_n(m0,m1,simd_width,offset_mod_simd);

            let (part, new_m)                                       
                = S::extract_elements(m0,m1,simd_width,offset_mod_simd);
            sum_vector = S::sum_vectors(&sum_vector,&part);

            m0 = new_m;
            // x0  = xform.read_next_with_mask();
            // i0  = input.read_next_with_mask();
            x0 = S::read_next_with_mask(&mut xform_mod_index,
                              &mut xform_array_index,
                              xform_array,
                              xform_size,
                              &mut xform_ra_size,
                              &mut xform_ra,
                              &mut xform_mask);
            i0 = S::read_next_with_mask(&mut input_mod_index,
                              &mut input_array_index,
                              input_array,
                              input_size,
                              &mut input_ra_size,
                              &mut input_ra,
                              &mut input_mask);
            m1  = S::cross_product(x0,i0); // new m1
            dp_counter += simd_width;
            // offset_mod_simd unchanged
        }
        // above may have set dp_counter to k already.
        if dp_counter < k {            // If not, ...
            let want = k - dp_counter; // always strictly positive

            // eprintln!("Calling sum_across_n with m0 {:?}, m1 {:?}, n {}, offset {}",
            //      m0.vec, m1.vec, want, offset_mod_simd);
            // let (part, new_m) = S::sum_across_n(m0,m1,want,offset_mod_simd);

            let (part, new_m) = S::extract_elements(m0,m1,want,offset_mod_simd);
            
            // eprintln!("got sum {}, new m {:?}", part, new_m.vec);

            sum_vector = S::sum_vectors(&sum_vector,&part);
            if offset_mod_simd + want >= simd_width {
                // consumed m0 and maybe some of m1 too
                m0 = new_m;     // nothing left in old m0, so m0 <- m1
                // x0  = xform.read_next_with_mask();
                // i0  = input.read_next_with_mask();
                x0 = S::read_next_with_mask(&mut xform_mod_index,
                                  &mut xform_array_index,
                                  xform_array,
                                  xform_size,
                                  &mut xform_ra_size,
                                  &mut xform_ra,
                                  &mut xform_mask);
                i0 = S::read_next_with_mask(&mut input_mod_index,
                                  &mut input_array_index,
                                  input_array,
                                  input_size,
                                  &mut input_ra_size,
                                  &mut input_ra,
                                  &mut input_mask);
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

        let sum = S::sum_across_simd(sum_vector);

        // handle writing and incrementing or, oc
        let write_index = or * down + oc * right;
        output.indexed_write(write_index,sum);
        or = if or + 1 < orows { or + 1 } else { 0 };
        oc = if oc + 1 < ocols { oc + 1 } else { 0 };

        sum_vector = zero;
        dp_counter = 0;
        total_dps += 1;
    }
}





// TODO:
//
// define some things as macros so that, eg, we can implement
// in-register storage of matrix contents.
//
// * Different types of masks
// * in-register storage of matrix
// * constants for lookup tables
// 

#[derive(Debug,Copy,Clone)]
pub struct VmullEngine8x8 {
    // using uint8x8_t rather than poly8x8_t since it involves less
    // type conversion.
    vec : uint8x8_t,
}

// low-level intrinsics
impl VmullEngine8x8 {

    #[inline(always)]
    unsafe fn read_simd(ptr: *const u8) -> Self {
        vld1_p8(ptr).into()
    }

    // unsafe fn read_simd_uint(ptr: *const u8) -> uint8x8_t {
    //  vld1_u8(ptr)
    // }

    #[allow(unused)]
    unsafe fn rotate_right(v : Self, amount : usize) -> Self {
        let mut mask = transmute( [0u8,1,2,3,4,5,6,7] ); // null rotate mask
        let add_amount = vmov_n_u8(amount as u8);
        let range_mask = vmov_n_u8(0x07);
        mask = vadd_u8(mask, add_amount);
        mask = vand_u8(mask, range_mask);
        vtbl1_u8(v.vec, mask).into()
    }

    #[allow(unused)]
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
    #[inline(always)]
    unsafe fn extract_from_offset(lo: &Self, hi : &Self, offset : usize)
                                  -> Self {
        debug_assert!(offset < 8);
        let tbl2 = uint8x8x2_t ( lo.vec, hi.vec );
        let mut mask = transmute( [0u8,1,2,3,4,5,6,7] ); // null rotate mask
        let add_amount = vmov_n_u8(offset as u8);
        mask = vadd_u8(mask, add_amount);
        vtbl2_u8(tbl2, mask).into()     
    }

    // The most common path through read_next involves calling
    // extract_from_offset(&ra,&r0, 8 - *ra_size). If we remember the
    // mask and pass it in each time instead, we should see some
    // performance improvement over calculating it from scratch each
    // time. Obviously, if ra_size changes, the mask also needs to
    // change.
    #[inline(always)]
    unsafe fn extract_using_mask(lo: &Self, hi : &Self, mask : &Self)
                                  -> Self {
        let tbl2 = uint8x8x2_t ( lo.vec, hi.vec );
        // 3 vector instructions saved: load mask, splat off, add
        vtbl2_u8(tbl2, mask.vec).into() 
    }

    // create mask, called when starting off and *ra_size changes
    unsafe fn extract_mask_from_offset(offset : usize) -> Self {
        // debug_assert!(offset < 8);
        let mask = transmute( [0u8,1,2,3,4,5,6,7] ); // null rotate mask
        let add_amount = vmov_n_u8(offset as u8);
        vadd_u8(mask, add_amount).into()
    }
    
    #[allow(unused)]
    unsafe fn splat(elem : u8) -> Self {
        vmov_n_u8(elem).into()
    }
    
    #[allow(unused)]
    unsafe fn mask_start_elements(v : Self, count : usize) -> Self {
        debug_assert!(count > 0);
        let mask = Self::shift_right(Self::splat(0xff),
                                     (8usize - count).into());
        vand_u8(v.vec, mask.vec).into() 
    }
    
    #[allow(unused)]
    unsafe fn mask_end_elements(v : Self, count : usize) -> Self {
        debug_assert!(count > 0);
        let mask = Self::shift_left(Self::splat(0xff),
                                    (8usize - count).into());
        vand_u8(v.vec, mask.vec).into() 
    }

    // no need for negated forms of the above because
    // negative_mask_start_elements(v,x) would be the same as
    // mask_end_elements(v,8-x).

    #[allow(unused)]
    unsafe fn non_wrapping_read(read_ptr :  *const u8,
                                beyond   :  *const u8
    ) -> Option<Self> {
        if read_ptr.offset(Self::SIMD_BYTES as isize) > beyond {
            None
        } else {
            Some(Self::read_simd(read_ptr).into())
        }
    }

    #[allow(unused)]
    unsafe fn wrapping_read(read_ptr : *const u8,
                            beyond   : *const u8,
                            restart  : *const u8
    ) -> (Self, Option<Self>) {

        let missing : isize
            = (read_ptr.offset(Self::SIMD_BYTES as isize)).offset_from(beyond);
        debug_assert!(missing >= 0);

        // get 8 - missing from end of stream
        let mut r0 = Self::read_simd(read_ptr);

        // 
        if missing == 0 {
            return (r0.into(), None);
        }

        // get missing from start of stream
        let r1 = Self::read_simd(restart);

        // Two steps to combine...
        // * shift r0 left by missing (move bytes to top)
        // * extract 8 bytes from {r1:r0} at offset (8 -missing)
        r0 = Self::shift_left(r0.into(), missing as usize);
        r0 = Self::extract_from_offset(&r0, &r1, 8-missing as usize);

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

        // One more thought ... this Simd engine is based on 64-bit
        // registers, but the only part that actually needs to be
        // 64-bit is the multiply routine (thanks to vmull being a
        // widening operation). At least, I think that's the case.
        // Maybe some vtbl instructions are missing?
        //
        // Anyhow, if the multiply is the only thing that is
        // restricted, I can drop down and do two 64-by-64
        // multiplications when needed, but continue to use 128-wide
        // instructions elsewhere. I'd have to copy/paste a lot of
        // code and edit it (as opposed to trying to make a generic
        // version) because the mnemonics/intrinsics are going to be
        // different. Although, I could add another (thin) translation
        // layer to map general names like load_u8_data (better:
        // load_elements) to to the appropriate 64-bit/128-bit load
        // intrinsics.

        // Anyway, back to where I left off:
        
        // To get updated r1 (with missing bytes removed), either:
        // * right shift by missing
        // * mask out lower missing bytes

        // I'm going to go with masking, so the read-ahead will always
        // be in the high bytes of the vector.
        (r0, Some(Self::mask_end_elements(r1, 8 - missing as usize)))
    }
    
}

// Type conversion is very useful
impl From<uint8x8_t> for VmullEngine8x8 {
    fn from(other : uint8x8_t) -> Self {
        Self { vec : other }
    }
}
impl From<poly8x8_t> for VmullEngine8x8 {
    fn from(other : poly8x8_t) -> Self {
        unsafe {
            Self { vec : vreinterpret_u8_p8(other) }
        }
    }
}


// impl ArmSimd for VmullEngine8x8 {
impl Simd for VmullEngine8x8 {
    type V = uint8x8_t;
    type E = u8;
    const SIMD_BYTES : usize = 8;

    #[inline(always)]
    fn zero_element() -> Self::E { 0 }
    #[inline(always)]
    fn add_elements(a : Self::E, b : Self::E) -> Self::E { (a ^ b).into() }

    #[inline(always)]
    fn zero_vector() -> Self {
        unsafe { vmov_n_u8(0).into() }
    }

    unsafe fn starting_mask() -> Self {
        Self::read_simd(vec![8,9,10,11,12,13,14,15].as_ptr())
    }

    // Keep values to be summed in vector for as long as possible
    #[inline(always)]
    fn sum_vectors(a : &Self, b : &Self) -> Self {
        unsafe {
            let a : uint64x1_t = vreinterpret_u64_u8(a.vec);
            let b : uint64x1_t = vreinterpret_u64_u8(b.vec);
            vreinterpret_u8_u64(veor_u64(a, b)).into()
        }
    }

    fn sum_across_simd(v : Self) -> u8 {
        unsafe {
            let mut v : uint64x1_t = vreinterpret_u64_u8(v.vec);
            // it seems that n is bits? No. Bytes. No, it's bits after all.
            // eprintln!("Starting v: {:x?}", v);

            v = veor_u64(v, vshr_n_u64::<32>(v));// eprintln!("after shift 4: {:x?}", v);
            v = veor_u64(v, vshr_n_u64::<16>(v));// eprintln!("after shift 2: {:x?}", v);
            v = veor_u64(v, vshr_n_u64::<8>(v)); // eprintln!("after shift 1: {:x?}", v);
            let ret = vget_lane_u8::<0>(vreinterpret_u8_u64(v));
            // eprintln!("sum_across_simd returning: {:x}", ret);
            return ret;

            // There might be an alternative way, using re-casting
            //
            // If v is in a D reg (64 bits), we might be able to refer to
            // its two 32-bit halves? Ah, no... no "across"-like
            // intrinsics, it seems.
        }            
    }

    fn extract_elements(lo : Self, hi : Self, n : usize, off : usize)
                           -> (Self, Self) {
        // let m = if off + n >= 8 { hi } else { lo };
        unsafe {        
            // can use left and right shifts, which might be more
            // efficient
            let mut bytes = lo;
            if off > 0 {
                bytes = Self::shift_right(bytes, off);
            }
            if off + n >= 8 {
                // let extracted = Self::extract_from_offset(&lo, &hi, off);
                // let masked = Self::mask_start_elements(extracted, n).into();
                // let result = Self::sum_across_simd(masked);

                // if we need some from hi
                if off + n > 8 {
                    bytes = veor_u8(bytes.vec,
                                    Self::shift_left(hi, 16 - (off + n)).vec)
                        .into();
                }

                // eprintln!("Got lo: {:x?}, hi: {:x?}, n: {}, off: {}",
                //        lo.vec, hi.vec, n, off);
                // eprintln!("extracted: {:x?}", extracted.vec);
                // eprintln!("masked: {:x?}", masked.vec);
                // eprintln!("xor result: {:x}", result);

                ( bytes, hi )
            } else {

                if n < 8 {
                    bytes = Self::shift_left(bytes, 8 - n);
                }

                (bytes,lo)

                // eprintln!("Got lo: {:x?}, hi: {:x?}, n: {}, off: {}",
                //        lo.vec, hi.vec, n, off);
                // eprintln!("extracted: {:x?}", extracted.vec);
                // eprintln!("masked: {:x?}", masked.vec);
                // eprintln!("xor result: {:x}", result);
            }

        }
    }


    // #[inline(always)]
    fn cross_product(a : Self, b : Self) -> Self {
        unsafe {
            simd_mull_reduce_poly8x8(&vreinterpret_p8_u8(a.vec),
                                     &vreinterpret_p8_u8(b.vec)).into()
        }
    }

    /// load from memory (useful for testing, benchmarking)
    unsafe fn from_ptr(ptr: *const Self::E) -> Self {
        Self::read_simd(ptr)
    }

    /// Cross product of two slices; useful for testing, benchmarking
    /// Uses fixed poly at the moment.
    fn cross_product_slices(dest: &mut [Self::E],
                            av : &[Self::E], bv : &[Self::E]) {

        debug_assert_eq!(av.len(), bv.len());
        debug_assert_eq!(bv.len(), dest.len());

        let bytes = av.len();
        if bytes & 7 != 0 {
            panic!("Buffer length not a multiple of 8");
        }
        let mut times = bytes >> 3;

        // convert dest, av, bv to pointers
        let mut dest = dest.as_mut_ptr();
        let mut av = av.as_ptr();
        let mut bv = bv.as_ptr();

        while times > 0 {
            times -= 1;
            let a : Self;
            let b : Self;
            let res  : Self;

            // read in a, b from memory
            unsafe { 
                a = Self::read_simd(av); // 
                b = Self::read_simd(bv); // b = *bv also crashes
                av = av.offset(1);      // offset for vec type, not bytes!
                bv = bv.offset(1);
            }

            res =  Self::cross_product(a, b);

            unsafe {
                vst1_u8(dest, res.vec);
                dest = dest.offset(1);
            }
        }
    }

    // caller passes in current state variables and we pass back
    // updated values plus the newly-read simd value
    // #[inline(always)]
    unsafe fn read_next(mod_index : &mut usize,
                        array_index : &mut usize,
                        array     : &[Self::E],
                        size      : usize,
                        ra_size   : &mut usize,
                        ra        : &mut Self)
                        -> Self {

        let new_ra : Self; // = r0; // silence compiler
        let mut new_mod_index = *mod_index;
        let new_ra_size; //   = *ra_size;

        // Both should be OK, but it's safer to only look at
        // mod_index, since that only changes once per call and so is
        // easier to reason about:

        let available_at_end = size - *array_index;
        // apparently not:
        // let available_at_end = size - *mod_index;

        let available = *ra_size + available_at_end;

        // at end, when updating mod_index, we wrap it around, so this
        // should always be strictly positive
        debug_assert!(available_at_end > 0);

        // eprintln!("Starting mod_index: {}", *mod_index);
        // eprintln!("Starting array_index: {}", *array_index);
        // eprintln!("Available: ra {} + at end {} = {}",
        //        *ra_size, available_at_end, available);

        // r0 could have full or partial simd in it
        let read_ptr = array.as_ptr().offset((*array_index) as isize);
        let mut r0 : Self = Self::read_simd(read_ptr as *const u8).into();
        *array_index += 8;

        // eprintln!("Read r0: {:x?}", r0.vec);

        let result;
        // let mut have_r1 = false;
        let r1; // = r0;                // silence compiler

        // Check against array_index here because we've incremented
        // it. Could also check if available_at_end <= 8, which should
        // be the same thing.
        // let array_bool = *array_index >= size;
        // let avail_bool = available_at_end <= 8;

        // apparently not...
        // debug_assert_eq!(array_bool, avail_bool);

        // restructure if/then to put most common case first, then
        // return immediately without jumping to end
        if *array_index < size {  // *array_index < size

            // eprintln!("*array_index < size");

            // start with the easiest (and most common) case:
            if *ra_size > 0 {
                // combine ra + r0.
                result = Self::extract_from_offset(&ra, &r0, 8 - *ra_size);
            } else {
                // else all 8 bytes come from r0
                result = r0;
            }
            new_ra = r0;

            // save updated variables
            //      *ra_size = new_ra_size;
            *ra = new_ra;

            new_mod_index += 8;
            if new_mod_index >= size { new_mod_index -= size }
            *mod_index = new_mod_index;

            // eprintln!("final ra_size: {}", *ra_size);
            // eprintln!("final ra: {:x?}", (*ra).vec);
            // eprintln!("result: {:x?}", result.vec);
            
            if *array_index >= size {
                // not an error
                // eprintln!("Fixing up array index {} to zero", *array_index);
                *array_index = 0;
            }

            return result;

        } else  { // *array_index >= size {

            // eprintln!("*array_index >= size");
            
            // This means that r0 is the last read from the array.

            // Scenarios for reading:
            //
            // a) There's not enough available between ra + r0, so we
            //    have to read from the start of the stream again
            //
            // b) We have enough between ra + r0 (so no read)
            //
            // Scenario a will definitely produce readahead (since
            // we're reading 8 new values from the start), while
            // scenario b may have readahead depeding on the
            // comparison of available <=> 8
            //
            // Obviously new readahead size calculations will be
            // different for each.

            if available < 8 {

                // eprintln!("*array_index >= size && available < 8");

                let read_ptr = array.as_ptr();
                r1 = Self::read_simd(read_ptr as *const u8);
                *array_index = 8;

                // We had `available` from ra and r0, and we read
                // 8, then returned 8, so we still have `available`

                // eprintln!("Changing ra_size to available");
                new_ra_size = available;

                // how best to approach register shifting depends on
                // whether we have ra or not.

                if *ra_size > 0 {

                    // Combine ra and r0

                    // ra is already in top, so extract_from_offset
                    // works fine with r0

                    r0 = Self::extract_from_offset(&ra, &r0, 8 - *ra_size);

                    // now we have available bytes in r0, so to use
                    // extract_from_offset again with r0, r1, we have
                    // to move those bytes to the top
                    r0 = Self::shift_left(r0, 8 - available);
                    result = Self::extract_from_offset(&r0, &r1, 8 - available);

                    // r1 already has its bytes in the right place (at top)
                    new_ra = r1;                    

                } else {

                    // no readahead, so just combine bytes of r0, r1
                    r0 = Self::shift_left(r0, 8 - available);
                    result = Self::extract_from_offset(&r0, &r1, 8 - available);
                    new_ra = r1;                    
                }                   
                
            } else {  // array_index >= size && available >= 8

                // eprintln!("*array_index >= size && available >= 8");

                // Scenario b for end of stream (no read)

                // new_ra_size is 8 less because we take 8 without
                // replenishing with another read
                new_ra_size = available - 8;

                // if r0 still has bytes after this, we have to shift
                // them to the top as new ra

                // last bug (until the next one):
                
                if new_ra_size > 0 {
                    // new_ra = Self::shift_left(r0,8 - *ra_size);
                    new_ra = Self::shift_left(r0,8 - available_at_end);
                } else {
                    // value is junk, so we don't need to write
                    new_ra = r0; // keep compiler happy
                }

                // here, too, we check whether we have readahead and
                // do different register ops depending
                if *ra_size > 0 {
                    // combine ra + r0.
                    result = Self::extract_from_offset(&ra, &r0, 8 - *ra_size);
                } else {
                    // else all 8 bytes come from r0
                    result = r0;
                }
            }
            
        }

        // save updated variables

        *ra_size = new_ra_size;
        *ra = new_ra;

        new_mod_index += 8;
        if new_mod_index >= size { new_mod_index -= size }
        *mod_index = new_mod_index;

        // eprintln!("final ra_size: {}", *ra_size);
        // eprintln!("final ra: {:x?}", (*ra).vec);
        // eprintln!("result: {:x?}", result.vec);
        
        if *array_index >= size {
            // not an error
            // eprintln!("Fixing up array index {} to zero", *array_index);
            *array_index = 0;
        }

        return result;

        // END REWRITE!
        
    }

    //#[inline(always)]
    unsafe fn read_next_with_mask(mod_index : &mut usize,
                                  array_index : &mut usize,
                                  array     : &[Self::E],
                                  size      : usize,
                                  ra_size   : &mut usize,
                                  ra        : &mut Self,
                                  mask : &mut Self)
                        -> Self {

        let new_ra : Self; // = r0; // silence compiler
        let mut new_mod_index = *mod_index;
        let new_ra_size; //   = *ra_size;

        // Both should be OK, but it's safer to only look at
        // mod_index, since that only changes once per call and so is
        // easier to reason about:

        let available_at_end = size - *array_index;
        // apparently not:
        // let available_at_end = size - *mod_index;

        let available = *ra_size + available_at_end;

        // at end, when updating mod_index, we wrap it around, so this
        // should always be strictly positive
        debug_assert!(available_at_end > 0);

        // eprintln!("Starting mod_index: {}", *mod_index);
        // eprintln!("Starting array_index: {}", *array_index);
        // eprintln!("Available: ra {} + at end {} = {}",
        //        *ra_size, available_at_end, available);

        // r0 could have full or partial simd in it
        let read_ptr = array.as_ptr().offset((*array_index) as isize);
        let mut r0 : Self = Self::read_simd(read_ptr as *const u8).into();
        *array_index += 8;

        // eprintln!("Read r0: {:x?}", r0.vec);

        let result;
        // let mut have_r1 = false;
        let r1; // = r0; // silence compiler

        // Check against array_index here because we've incremented
        // it. Could also check if available_at_end <= 8, which should
        // be the same thing.
        // let array_bool = *array_index >= size;
        // let avail_bool = available_at_end <= 8;

        // apparently not...
        // debug_assert_eq!(array_bool, avail_bool);

        // restructure if/then to put most common case first, then
        // return immediately without jumping to end
        if *array_index < size {  // *array_index < size

            // eprintln!("*array_index < size");

            // start with the easiest (and most common) case:
            if false {
                if *ra_size > 0 {
                    // combine ra + r0.
                    result = Self
                    // ::extract_from_offset(&ra, &r0, 8 - *ra_size);
                        ::extract_using_mask(&ra, &r0, &*mask);
                } else {
                    // else all 8 bytes come from r0
                    result = r0;
                }
            } else {
                
                // always call intrinsic
                result = Self
                // ::extract_from_offset(&ra, &r0, 8 - *ra_size);
                    ::extract_using_mask(&ra, &r0, &*mask);

            }
            new_ra = r0;

            // save updated variables
            //      *ra_size = new_ra_size;
            *ra = new_ra;

            new_mod_index += 8;
            if new_mod_index >= size { new_mod_index -= size }
            *mod_index = new_mod_index;

            // eprintln!("final ra_size: {}", *ra_size);
            // eprintln!("final ra: {:x?}", (*ra).vec);
            // eprintln!("result: {:x?}", result.vec);
            
            if *array_index >= size {
                // not an error
                // eprintln!("Fixing up array index {} to zero", *array_index);
                *array_index = 0;
            }

            return result;

        } else  { // *array_index >= size {

            // eprintln!("*array_index >= size");
            
            // This means that r0 is the last read from the array.

            // Scenarios for reading:
            //
            // a) There's not enough available between ra + r0, so we
            //    have to read from the start of the stream again
            //
            // b) We have enough between ra + r0 (so no read)
            //
            // Scenario a will definitely produce readahead (since
            // we're reading 8 new values from the start), while
            // scenario b may have readahead depeding on the
            // comparison of available <=> 8
            //
            // Obviously new readahead size calculations will be
            // different for each.


            // Rework: repeat elements from the start of the matrix
            // after its end. This eliminates the need to combine
            // bytes read from the end of the stream and the start.
            
            if available < 8 {

                // eprintln!("*array_index >= size && available < 8");

                let read_ptr = array.as_ptr();
                r1 = Self::read_simd(read_ptr as *const u8);
                *array_index = 8;

                // We had `available` from ra and r0, and we read
                // 8, then returned 8, so we still have `available`

                // eprintln!("Changing ra_size to available");
                new_ra_size = available;

                // how best to approach register shifting depends on
                // whether we have ra or not.

                let new_mask = Self::extract_mask_from_offset(8 - available);

                if false {
                    if *ra_size > 0 {

                        /* old code
                        panic!();
                        // Combine ra and r0

                        // ra is already in top, so extract_from_offset
                        // works fine with r0
                        
                        r0 = Self
                        //::extract_from_offset(&ra, &r0, 8 - *ra_size);
                        ::extract_using_mask(&ra, &r0, &*mask);

                        // now we have available bytes in r0, so to use
                        // extract_from_offset again with r0, r1, we have
                        // to move those bytes to the top
                        r0 = Self::shift_left(r0, 8 - available);
                        result = Self::extract_using_mask(&r0, &r1, &new_mask);

                        // r1 already has its bytes in the right place (at top)
                        new_ra = r1;                    

                         */

                        // new code

                        // still have to combine ra, r0
                        r0 = Self
                        //::extract_from_offset(&ra, &r0, 8 - *ra_size);
                            ::extract_using_mask(&ra, &r0, &*mask);

                        // but no shift/extract to combine r0, r1
                        result = r0;
                        new_ra = r1

                    } else {

                        /*
                        // no readahead, so just combine bytes of r0, r1
                        r0 = Self::shift_left(r0, 8 - available);
                        result = Self::extract_using_mask(&r0, &r1, &new_mask);
                        new_ra = r1;
                         */

                        // again, no need to combine r0, r1
                        result = r0;
                        new_ra = r1;
                    }
                } else {
                        r0 = Self
                        //::extract_from_offset(&ra, &r0, 8 - *ra_size);
                            ::extract_using_mask(&ra, &r0, &*mask);

                        // but no shift/extract to combine r0, r1
                        result = r0;
                        new_ra = r1
                }
                *mask = new_mask;

            } else {  // array_index >= size && available >= 8

                // eprintln!("*array_index >= size && available >= 8");

                // Scenario b for end of stream (no read)

                // new_ra_size is 8 less because we take 8 without
                // replenishing with another read
                new_ra_size = available - 8;

                // if r0 still has bytes after this, we have to shift
                // them to the top as new ra

                // last bug (until the next one):

                if false {
                    if new_ra_size > 0 {
                        // new_ra = Self::shift_left(r0,8 - *ra_size);
                        new_ra = Self::shift_left(r0,8 - available_at_end);
                    } else {
                        // value is junk, so we don't need to write
                        new_ra = r0; // keep compiler happy
                    }
                } else {
                    new_ra = Self::shift_left(r0,8 - available_at_end);
                }                    

                // here, too, we check whether we have readahead and
                // do different register ops depending
                if false {
                    if *ra_size > 0 {
                        // combine ra + r0.
                        result = Self
                        // ::extract_from_offset(&ra, &r0, 8 - *ra_size);
                            ::extract_using_mask(&ra, &r0, &*mask);
                    } else {
                        // else all 8 bytes come from r0
                        result = r0;
                    }
                } else {
                    result = Self
                    // ::extract_from_offset(&ra, &r0, 8 - *ra_size);
                        ::extract_using_mask(&ra, &r0, &*mask);

                }
                *mask = Self::extract_mask_from_offset(8 - new_ra_size);
            }
        }

        // save updated variables

        *ra_size = new_ra_size;
        *ra = new_ra;

        new_mod_index += 8;
        if new_mod_index >= size { new_mod_index -= size }
        *mod_index = new_mod_index;

        // eprintln!("final ra_size: {}", *ra_size);
        // eprintln!("final ra: {:x?}", (*ra).vec);
        // eprintln!("result: {:x?}", result.vec);
        
        if *array_index >= size {
            // not an error
            // eprintln!("Fixing up array index {} to zero", *array_index);
            *array_index = 0;
        }

        return result;

        // END REWRITE!
        
    }
                        
    // Sum across N
    //
    // There are various ways to do this...
    //
    // Given a pair of registers, if the area being extracted
    // straddles the two, things are a bit tricky using only rotating
    // masks.
    //
    // We have m0, m1
    //
    // m0 is easy enough:
    //
    // have a mask with n bytes set to extract from the start of m0.
    //
    // apply the mask to pull out the bytes of interest, then sum them
    // (optionally) apply the reverse mask to blank low bytes)
    // shift the mask to the right (or rotate, if low bytes were blanked)
    //
    // When the mask rotates beyond the end, it will have some bytes
    // at the start and some bytes at the end. We can save the initial
    // mask and apply it to the wrapping mask so that only the low
    // bytes will be selected.
    //
    // eg, n = 7
    //
    // initial mask  1111_1110 0000_0000   select bytes 0..6 of m0
    // after ror 7:  0000_0001 1111_1100
    // after ror 14: 1111_1000 0000_0011
    //
    // after advancing beyond m0, we will read a new m2, and rotate
    // the mask by 8 (swap big, small), giving:
    //
    //                   m0        m1        m2 ...
    // initial mask  1111_1110 0000_0000            select bytes 0..6 of m0
    // after ror 7:  0000_0001 1111_1100            select across m0, m1, advance
    // after ror 14:           0000_0011 1111_1000  select across m1, m2, advance

    // The same mask can be used repeatedly.

    // IIRC, on armv7, we can't use a 16-wide register as input to
    // vtbl, but we can use an 8-wide register to look up from two
    // 8-wide registers.

    // With all that said, to start with, just use the existing
    // extract_from_offset routine and use a non-rotating mask to
    // extract the first n elements from the returned 8-wide vector.
    // Worry about efficiency (not recalculating masks and rotates
    // every time we call it) later.

    unsafe fn sum_across_n(lo : Self, hi : Self, n : usize, off : usize)
                           -> (Self::E, Self) {
        // let m = if off + n >= 8 { hi } else { lo };
        
        // can use left and right shifts, which might be more
        // efficient
        let mut bytes = lo;
        if off > 0 {
            bytes = Self::shift_right(bytes, off);
        }
        if off + n >= 8 {
            // let extracted = Self::extract_from_offset(&lo, &hi, off);
            // let masked = Self::mask_start_elements(extracted, n).into();
            // let result = Self::sum_across_simd(masked);

            // if we need some from hi
            if off + n > 8 {
                bytes = veor_u8(bytes.vec,
                                 Self::shift_left(hi, 16 - (off + n)).vec)
                    .into();
            }

            // eprintln!("Got lo: {:x?}, hi: {:x?}, n: {}, off: {}",
            //        lo.vec, hi.vec, n, off);
            // eprintln!("extracted: {:x?}", extracted.vec);
            // eprintln!("masked: {:x?}", masked.vec);
            // eprintln!("xor result: {:x}", result);

            ( Self::sum_across_simd(bytes), hi )
        } else {

            if n < 8 {
                bytes = Self::shift_left(bytes, 8 - n);
            }

            (Self::sum_across_simd(bytes),lo)

            // eprintln!("Got lo: {:x?}, hi: {:x?}, n: {}, off: {}",
            //        lo.vec, hi.vec, n, off);
            // eprintln!("extracted: {:x?}", extracted.vec);
            // eprintln!("masked: {:x?}", masked.vec);
            // eprintln!("xor result: {:x}", result);
        }
    }

}


// Interleaving C version in comments

// void simd_mull_reduce_poly8x8(poly8x8_t *result,
//                            poly8x8_t *a, poly8x8_t *b) {

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
        //      vst1_u8(result, narrowed);
        vreinterpret_p8_u8(narrowed)
    }
}

// In-register storage of matrix
//
// If the entire matrix will fit in n simd registers, we can use n+1
// registers to achieve easy lookup/rotation.
//
// Say that the stream is 3 bytes, and we store it in a single
// register:
//
// abcabcab
//
// We can't rotate that vector to get a new position in the stream
// because 3 is relatively prime to 8 (we'd get invalid data like
// 'aba' at some point).
//
// We can, however, use two registers:
//
// abcabcab cabcabca (high)
//
// Then we can have a mod-3 offset pointer returning all possible
// reads:
//
// abcabcab cabcabca (high)
//   cab... ca       offset 2
//  bca...  c        offset 1
// abc...            offset 0
//
// This can be fairly easily extended to multi-register schemes.
//
// Also, if the stream length is a multiple of the simd length, we can
// avoid storing an extra register, eg, with a mod-4 counter:
//
// r0       r1
// abcdabcd abcdacd
//    dab.. abc      offset 3
//   cab... ab       offset 2
//  bca...  a        offset 1
// abc...            offset 0
//
//
// Since r0 = r1, we just use extract_from_offset(&r0,&r0) instead of
// explicitly storing r1.
//
// (state stored within matrix multiply routine)


/// Matrix storage type for Arm
///
pub struct ArmMatrix<S : Simd> {

    // set up a dummy value as an alternative to PhantomData
    _zero: S,

    // to implement regular matrix stuff
    rows : usize,
    cols : usize,
    pub array : Vec<u8>,
    is_rowwise : bool,
}

/// Concrete implementation of matrix for Arm
impl ArmMatrix<VmullEngine8x8> {

    pub fn fill(&mut self, data : &[u8]) {
        let size = self.size();
        if data.len() != size {
            panic!("Supplied {} data bytes  != matrix size {}",
            data.len(), size);
        }
        self.array[0..size].copy_from_slice(data);
        // make read-around work:
        let mut index = 0;
        while index <  7 {
            self.array[size + index] = data[index % size];
            index += 1;
        }
    }

    pub fn new_with_data(rows : usize, cols : usize, is_rowwise : bool,
                     data : &[u8]) -> Self {
        let mut this = Self::new(rows, cols, is_rowwise);
        this.fill(data);
        this
    }

}


use guff::F8;

impl SimdMatrix<VmullEngine8x8,F8> for ArmMatrix<VmullEngine8x8> {

    fn new(rows : usize, cols : usize, is_rowwise : bool) -> Self {
        let size = rows * cols;
        // if size < 8 {
        //     panic!("This matrix can't handle rows * cols < 8 bytes");
        // }

        // add an extra 7 guard bytes beyond size
        let array = vec![0u8; size + 7];

        // set up a dummy value as an alternative to PhantomData
        let _zero = VmullEngine8x8::zero_vector();
        
        ArmMatrix::<VmullEngine8x8> {
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

    // #[inline(always)]
    fn indexed_write(&mut self, index : usize, elem : u8) {
        unsafe {                // genuinely unsafe
            let addr = self.array.as_mut_ptr().offset(index as isize);
            *addr = elem;
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        let size = self.size();
        &mut self.array[0..size]
    }
}


// Aligned reads mean that we have to take care of readahead and our
// wrapping code gets messy.
//
// Implement some low-level functions that can be used to test and
// benchmark doing wrap-around reads with non-aligned memory
// addresses.
//
// ARM NEON doesn't seem to have separate aligned/non-aligned
// load/store instructions, so we just use the regular ones.


// I'm not sure if there's a difference between using a purely
// functional style and passing mutable args. I'll try both and see
// what the benchmarks tell me.

pub fn nar_read_next_tuple(index : usize, size : usize, vec : &[u8])
                       -> (usize, VmullEngine8x8) {
    unsafe {
        let addr = vec.as_ptr();
        let ret = VmullEngine8x8::read_simd(addr.offset(index as isize));
        let mut new_index = index + 8;
        if new_index >= size {
            new_index -= size;
        }
        (new_index, ret)
    }
}

pub fn nar_read_next_mut(index : &mut usize, size : usize, vec : &[u8])
                     -> VmullEngine8x8 {
    unsafe {
        let addr = vec.as_ptr();
        let ret = VmullEngine8x8::read_simd(addr.offset(*index as isize));
        let mut new_index = *index + 8;
        if new_index >= size {
            new_index -= size;
        }
        *index = new_index;
        ret
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
        let mut _r : poly8x8_t;

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
    fn test_sum_across_simd() {
        unsafe {
            let data : uint8x8_t = transmute([1u8, 2,4,8,16,32,64,128]);
            let got = VmullEngine8x8::sum_across_simd(data.into());

            assert_eq!(255, got);

            let data : uint8x8_t = transmute([0u8,1, 2,4,8,16,32,64]);
            let got = VmullEngine8x8::sum_across_simd(data.into());

            assert_eq!(0x7f, got);
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

            let res = VmullEngine8x8::extract_from_offset(&r0.into(), &r1.into(), 0);
            assert_eq!(format!("{:x?}", r0),
                       format!("{:x?}", res.vec));

            let res = VmullEngine8x8::extract_from_offset(&r0.into(), &r1.into(), 1);
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

    #[test]
    fn test_mask_end_elements() {
        unsafe {
            let input : uint8x8_t = transmute([42u8,42,42,42, 42,42,42,42]);
            let expect_1 : uint8x8_t = transmute([0u8 ,0 ,0 ,0 , 0 ,0 ,0 ,42 ]);
            let expect_2 : uint8x8_t = transmute([0u8 ,0 ,0 ,0 , 0 ,0 ,42,42 ]);
            let expect_3 : uint8x8_t = transmute([0u8 ,0 ,0 ,0 , 0 ,42,42,42 ]);
            let expect_7 : uint8x8_t = transmute([0u8,42,42,42,  42,42,42,42]);
            let expect_8 : uint8x8_t = transmute([42u8,42,42,42, 42,42,42,42]);

            let got = VmullEngine8x8::mask_end_elements(input.into(),1);
            assert_eq!(format!("{:x?}", expect_1),
                       format!("{:x?}", got.vec));

            let got = VmullEngine8x8::mask_end_elements(input.into(),2);
            assert_eq!(format!("{:x?}", expect_2),
                       format!("{:x?}", got.vec));

            let got = VmullEngine8x8::mask_end_elements(input.into(),3);
            assert_eq!(format!("{:x?}", expect_3),
                       format!("{:x?}", got.vec));

            let got = VmullEngine8x8::mask_end_elements(input.into(),7);
            assert_eq!(format!("{:x?}", expect_7),
                       format!("{:x?}", got.vec));

            let got = VmullEngine8x8::mask_end_elements(input.into(),8);
            assert_eq!(format!("{:x?}", expect_8),
                       format!("{:x?}", got.vec));
        }
    }

    // test reading/wrap-around read.  I won't do too much here. Just
    // enough to satisfy myself that the routines work as expected. I
    // will probably tweak things later to account for existing
    // read-ahead. And also to account for the boundary condition. I
    // think that I should stick to a two-register version, so that
    // may mean not reading from the start of the stream in all cases
    // (since readahead + trailing + new stream can exceed 2 vectors)

    // #[test]
    #[allow(unused)]
    fn test_non_wrapping_read() {
        unsafe {
            // actual vector data is 42s. The rest is just padding to
            // avoid unsafe memory reads
            let vector = vec![42u8,42,42,42, 42,42,42,42,
                              42u8,42,42,42, 42,42,42,42,
                              42u8,42,42,42, 0,0,0,0,
                              0,0,0,0      , 0,0,0,0 ];
            let mut pointer = vector.as_ptr();
            let beyond  = pointer.offset(20);

            // first two reads should return Some(data), so just
            // unwrap() to test.
            let _ = VmullEngine8x8::non_wrapping_read(
                pointer, beyond).unwrap();
            let _ = VmullEngine8x8::non_wrapping_read(
                pointer.offset(8), beyond).unwrap();
            
            match VmullEngine8x8::non_wrapping_read(pointer.offset(16), beyond) {
                None => { },
                _ => { panic!("Should have got back None"); }
            }
        }
    }

    // #[test]
    #[allow(unused)]
    fn test_wrapping_read() {
        unsafe {
            // actual vector data is non-zeros. The rest is just
            // padding to avoid unsafe memory reads
            let vector = vec![1u8,  2, 3, 4,  5, 6, 7, 8,
                              42u8,42,42,42, 42,42,42,42,
                              41u8,40,39,38, 0,0,0,0,
                              0,0,0,0      , 0,0,0,0 ];
            let mut pointer = vector.as_ptr();
            let beyond  = pointer.offset(20);

            // first two reads should return Some(data), so just
            // unwrap() to test.
            let _ = VmullEngine8x8::non_wrapping_read(
                pointer, beyond).unwrap();
            let _ = VmullEngine8x8::non_wrapping_read(
                pointer.offset(8), beyond).unwrap();
            
            let try_non_wrapping = VmullEngine8x8
                ::non_wrapping_read(pointer.offset(16), beyond);
            match VmullEngine8x8::non_wrapping_read(pointer.offset(16), beyond) {
                None => { },
                // same as last test
                _ => { panic!("Should have got back None"); }
            }

            // now we should try wrapping
            let (first, next)  = VmullEngine8x8
                ::wrapping_read(pointer.offset(16), beyond, pointer);

            // wrapped read
            let expect_first : uint8x8_t = transmute([41u8,40,39,38, 1,2,3,4 ]);
            // remainder of restarted read stored in high bytes
            let expect_next  : uint8x8_t = transmute([0u8,0,0,0,     5,6,7,8 ]);

            assert_eq!(format!("{:x?}", expect_first),
                       format!("{:x?}", first.vec));

            assert_eq!(format!("{:x?}", expect_next),
                       format!("{:x?}", next.unwrap().vec));
        }
    }

    // new read_next to replace non_wrapping_read and wrapping_read
    #[test]
    fn test_read_next_simple() {

        // state variables that read_next will update
        let mut ra;
        ra = VmullEngine8x8::zero_vector();
        let mut ra_size = 0;
        let mut mod_index = 0;
        let mut array_index = 0;
        let size = 24;
        let array = [ 0u8,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                      0,0,0,0,0,0,0,0]; // padding

        // additionally, we'll track non-modular index so that we
        // can check that index % size == mod_index
        let mut index = 0;

        // use iterators to make a reference stream
        let mut check = (0u8..24).cycle();
        let mut check_vec = [0u8; 8];
        let addr = check_vec.as_ptr();
        let _old_mod_index = 0;
        for _ in 0..42 {
            unsafe {
                // isn't there a quicker way to take 8 elements? Can't use
                // check.chunks()
                for i in 0..8 {
                    check_vec[i] = check.next().unwrap();
                }
                index += 8;
                let got = VmullEngine8x8
                    ::read_next(&mut mod_index,
                                &mut array_index,
                                &array[..],
                                size,
                                &mut ra_size,
                                &mut ra);
                assert_eq!(mod_index, index % size);

                
                
                let v = VmullEngine8x8::read_simd(addr);

                assert_eq!(format!("{:x?}", got.vec),
                           format!("{:x?}", v.vec));

            }
        }
    }
    
    #[test]
    fn test_read_next() {

        // state variables that read_next will update
        let mut ra;
        ra = VmullEngine8x8::zero_vector();
        let mut ra_size = 0;
        let mut mod_index = 0;
        let mut array_index = 0;
        let size = 21;
        let array = [ 0u8,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                      0,0,0,0,0,0,0,0]; // padding

        // additionally, we'll track non-modular index so that we
        // can check that index % size == mod_index
        let mut index = 0;

        // use iterators to make a reference stream
        let mut check = (0u8..21).cycle();
        let mut check_vec = [0u8; 8];
        let addr = check_vec.as_ptr();
        let _old_mod_index = 0;
        for _ in 0..42 {
            unsafe {
                // isn't there a quicker way to take 8 elements? Can't use
                // check.chunks()
                for i in 0..8 {
                    check_vec[i] = check.next().unwrap();
                }
                eprintln!("\nAbsolute index {}", index);
                index += 8;
                let got = VmullEngine8x8
                    ::read_next(&mut mod_index,
                                &mut array_index,
                                &array[..],
                                size,
                                &mut ra_size,
                                &mut ra);
                assert_eq!(mod_index, index % size);

                let v = VmullEngine8x8::read_simd(addr);

                assert_eq!(format!("{:x?}", got.vec),
                           format!("{:x?}", v.vec));

            }
        }
    }

    #[test]
    fn test_sum_across_n() {
        // first byte of av is stored in lowest memory location
        let a0 = [ 0u8,   1,  2,  4,  8, 16, 32,  64, ]; // 0b0111_1111 
        let a1 = [ 128u8, 0,  1,  2,  4,  8, 16,  32, ]; // 0b1011_1111
        let a2 = [ 1u8,   2,  4,  8, 16, 32, 64, 128, ];
        let a3 = [ 0u8,   1,  2,  4,  8, 16, 32,  64, ];

        unsafe {

            // convert 
            let a0 = VmullEngine8x8::read_simd(a0.as_ptr());
            let a1 = VmullEngine8x8::read_simd(a1.as_ptr());
            let _a2 = VmullEngine8x8::read_simd(a2.as_ptr());
            let _a3 = VmullEngine8x8::read_simd(a3.as_ptr());

            // simplest case 
            let (sum,_new_m) = VmullEngine8x8::sum_across_n(a0, a1, 8, 0);
            let expect : u8 = 0b0111_1111;
            eprintln!("expect {:x}", expect);
            assert_eq!(sum, expect);
        }
    }


    #[test]
    fn test_read_next_vs_nars() {

        // state variables that read_next will update
        let mut ra;
        ra = VmullEngine8x8::zero_vector();
        let mut ra_size = 0;
        let mut mod_index = 0;
        let mut array_index = 0;
        let size = 21;
        let array = [ 0u8,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                      0,1,2,3,4,5,6,7]; // padding

        // additionally, we'll track non-modular index so that we
        // can check that index % size == mod_index
        let mut index = 0;

        // use iterators to make a reference stream
        let mut check = (0u8..21).cycle();
        let mut check_vec = [0u8; 8];
        let addr = check_vec.as_ptr();
        let _old_mod_index = 0;

        // variables for use with two non-aligned read functions
        let mut nar_index_fn = 0;
        let mut nar_index_mut = 0;
        
        for _ in 0..42 {
            unsafe {
                // isn't there a quicker way to take 8 elements? Can't use
                // check.chunks()
                for i in 0..8 {
                    check_vec[i] = check.next().unwrap();
                }
                eprintln!("\nAbsolute index {}", index);
                index += 8;

                // check against read_next
                let got = VmullEngine8x8
                    ::read_next(&mut mod_index,
                                &mut array_index,
                                &array[..],
                                size,
                                &mut ra_size,
                                &mut ra);
                assert_eq!(mod_index, index % size);

                // reuse same v
                let v = VmullEngine8x8::read_simd(addr);

                assert_eq!(format!("{:x?}", got.vec),
                           format!("{:x?}", v.vec));

                // check against nar_read_next_tuple
                let (new_nar_index_fn, got)
                    = nar_read_next_tuple(nar_index_fn, size, &array[..]);
                assert_eq!(mod_index, new_nar_index_fn % size);
                nar_index_fn = new_nar_index_fn;

                assert_eq!(format!("{:x?}", got.vec),
                           format!("{:x?}", v.vec));

                // check against nar_read_next_mut
                let got = nar_read_next_mut(&mut nar_index_mut, size,
                                            &array[..]);
                assert_eq!(mod_index, nar_index_mut % size);

                assert_eq!(format!("{:x?}", got.vec),
                           format!("{:x?}", v.vec));

            }
        }
    }
    
}
