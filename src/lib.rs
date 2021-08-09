//! Fast SIMD matrix multiplication for finite fields
//!
//! This crate implements two things:
//!
//! 1. Fast SIMD-based multiplication of vectors of finite field
//!    elements (GF(2<sup>8</sup>) with the polynomial 0x11b)
//!
//! 2. A matrix multiplication routine based on achieving 100%
//! utilisation of the above
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
use num::{Zero, One};
// use core::mem::size_of;


// Only one x86 implementation, included automatically
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

// I want to emit assembly for these
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn _monomorph() {

    //    use crate::x86::*;

    #[inline(never)]
    fn inner_fn<S : Simd<E=u8> + Copy>(
        xform  : &mut impl SimdMatrix<S,F8>,
        input  : &mut impl SimdMatrix<S,F8>,
        output : &mut impl SimdMatrix<S,F8>)
    where S::E : Copy + Zero + One {
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
    let mut transform = // mut because of iterator
        Matrix::new(9,9,true);
    transform.fill(&identity[..]);
    
    // 17 is coprime to 9
    let mut input =
        Matrix::new(9,17,false);
    let vec : Vec<u8> = (1u8..=9 * 17).collect();
    input.fill(&vec[..]);

    let mut output =
        Matrix::new(9,17,false);

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


// I had wanted to have different matrix implementations that offered
// different ways of implementing read_next(). I have moved away from
// that goal by moving state out of matrices, though. It may be better
// to offer different multiply functions.
//
// My most immediate goal now is providing Matrix and multiply support
// for Arm. I don't want to just copy/paste code, but that might be
// the best solution for now. To support that, I'll define an
// arch-dependent Matrix type:

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod types {
    pub type NativeSimd = crate::x86::X86u8x16Long0x11b;
    pub type Matrix = crate::x86::X86Matrix<NativeSimd>;
}

#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"),
          feature = "arm_vmull"))]
pub mod types {
    pub type NativeSimd = crate::arm_vmull::VmullEngine8x8;
    pub type Matrix = crate::arm_vmull::ArmMatrix::<NativeSimd>;
}

pub use types::*;

// (actually, copy/paste worked with only type/simd_bytes changes, so
// I can work on making a more generic matrix later)

/// GCD and LCM functions
pub mod numbers;
pub use numbers::*;

/// SIMD support, based on `simulator` module
///
/// This trait will be in main module and will have to be implemented
/// for each architecture
pub trait Simd {
    type E : std::fmt::Display; // elemental type, eg u8
    // type EE;                 // for specifying fallback GaloisField
    // type SEE;                // ditto

    type V;                     // vector type, eg [u8; 8]
    const SIMD_BYTES : usize;

    fn zero_vector() -> Self;

    // Keep interface change simple while testing
    unsafe fn starting_mask() -> Self;

    // methods to replace sum_across_n
    fn sum_vectors(a : &Self, b : &Self) -> Self;
    fn sum_across_simd(v : Self) -> Self::E;
    fn extract_elements(lo : Self, hi : Self, n : usize, off : usize)
                        -> (Self, Self) where Self: Sized;
    
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

    unsafe fn read_next_with_mask(mod_index : &mut usize,
                                  array_index : &mut usize,
                                  array     : &[Self::E],
                                  size      : usize,
                                  ra_size : &mut usize,
                                  ra : &mut Self,
                                  mask : &mut Self,
    ) -> Self
    where Self : Sized;

    /// load from memory (useful for testing, benchmarking)
    unsafe fn from_ptr(ptr: *const Self::E) -> Self
        where Self : Sized;

    /// Cross product of two slices; useful for testing, benchmarking
    /// Uses fixed poly at the moment.
    fn cross_product_slices(dest: &mut [Self::E],
                            av : &[Self::E], bv : &[Self::E]);
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
pub trait SimdMatrix<S : Simd<E=G::E>, G : GaloisField>
where Self : Sized, S::E : PartialEq + Copy + Zero + One,

{
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
    // fn reset(&mut self);

    // Wrap-around read of matrix, returning a Simd vector type
    // 
    // unsafe fn read_next(&mut self) -> S; // moved to Simd
    
    // Wrap-around diagonal write of (output) matrix
    // fn write_next(&mut self, val : S::E); // moved to matrix mul

    fn indexed_read(&self, index : usize) -> S::E;
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

    // Other typical matrix stuff (all implemented as defaults here):
    //
    // * inversion
    // * adjoining (helps with inversion/solving)
    // * solver
    // * row/column operations (add scaled row to another)
    // * transposition
    // * interleaved load/store
    // * copy

    fn new(rows : usize, cols : usize, is_rowwise : bool) -> Self;

    fn adjoin_right(&self, other : &Self) -> Self
    where  {
        assert_eq!(self.rows(), other.rows());
        assert_eq!(self.is_rowwise(), other.is_rowwise());

        let mut this = Self::new(self.rows(), self.cols() + other.cols(),
                             self.is_rowwise());
        
        if self.is_rowwise() {
            let left_cols = self.cols();
            let right_cols = other.cols();
            let out_cols = left_cols + right_cols;

            let mut left_iter = self.as_slice().chunks(left_cols);
            let mut right_iter = other.as_slice().chunks(right_cols);

            let out_iter = this.as_mut_slice().chunks_mut(out_cols);
            for output_row in out_iter {
                let (left,right) = output_row.split_at_mut(left_cols);
                left.copy_from_slice(left_iter.next().unwrap());
                right.copy_from_slice(right_iter.next().unwrap());
            }
        } else {
            let (left,right) = this.as_mut_slice().split_at_mut(self.size());
            left.copy_from_slice(self.as_slice());
            right.copy_from_slice(other.as_slice());
        }
        
        this
    }

    fn identity(size : usize, is_rowwise : bool) -> Self {
        let mut this = Self::new(size, size, is_rowwise);
        let mut index = 0;
        for _ in 0..size {
            this.indexed_write(index, S::E::one());
            index += size + 1;
        }
        this
    }

    // Inversion doesn't use SIMD for now. 

    /// Gauss-Jordan inversion
    fn invert(&self, field : &G) -> Option<Self>
    {

        assert!(self.is_rowwise());
        assert_eq!(self.rows(), self.cols());
        // deleted, since type constraints guarantees G::E == S::E
        // assert_eq!((field.order() >> 3) as usize, size_of::<S::E>());

        // adjoin an identity matrix on the right
        let mut mat = self.adjoin_right(&Self::identity(self.rows(), true));

        eprintln!("adjoined matrix before inverse: {:x?}",
                  mat.as_slice());

        // store these in variables to satisfy the borrow checker
        let rows = mat.rows();
        let cols = mat.cols();

        let mut index = 0;      // index of diagonal element
        let mut rowsize = cols; // distance from diagonal to end
        for diag in 0..mat.rows() {

            eprintln!("\ndiagonal is {}", diag);
            eprintln!("index is {}", index);

            // If the matrix is invertible, there must be a non-zero
            // value somewhere in this column. If a zero occurs on the
            // diagonal, find a non-zero value below this row and swap
            // rows.
            if mat.indexed_read(index) == S::E::zero() {
                let mut found_row = false;
                let mut index_below = index + cols;
                for _row_below in diag + 1..rows {
                    if mat.indexed_read(index_below) != S::E::zero() {
                        found_row = true;
                        break;
                    }
                    index_below += cols;
                }
                // It should be pretty clear that if the caller tries
                // to unwrap the inverse matrix and gets none, that
                // it's not invertible. Yes?
                if !found_row { return None }
                // Now fight with the borrow checker (not really)
                let (_,slice)   = mat.as_mut_slice().split_at_mut(index);
                let (row1,row2) = slice.split_at_mut(index_below - index);
                let row1 = &mut row1[..rowsize];
                let row2 = &mut row2[..rowsize];
                row1.swap_with_slice(row2);
                // (since all mutable borrows are dropped here)
            }

            // Normalise the diagonal so that it becomes 1. I probably
            // need an in-place scale in main guff crate. Do I have
            // one already?


            // 困った！ .. need a type-based solution after all... I
            // can't get inv from an arbitrary field and have it be
            // the same as S::E... So, make Simd generic on both the
            // architecture-specific stuff and the underlying element
            // type, supplied by a concrete type from guff?
            // Hmm... more likely the Matrix implementation will be
            // keyed to both, although later on Simd will also depend
            // on a reference implementation of GF(2) due to needing
            // to build poly-specific lookup tables.
            let inverse = field.inv(mat.indexed_read(index));

            eprintln!("inverse of element {:x?} is {:x?}",
                      mat.indexed_read(index), inverse);

            mat.indexed_write(index, S::E::one()); // element/element
            field.vec_constant_scale_in_place(
                &mut mat.as_mut_slice()[index + 1 .. index + rowsize],
                inverse);
            eprintln!("matrix after normalising row {}: {:x?}",
                      diag,
                      // &mat.as_slice()[diag * cols .. diag * cols + cols]
                      &mat.as_slice()[ .. ]
            );
            
            // Scan up and down from the diagonal adding a multiple of
            // the current row so that the target row has zero in this
            // column. Need a mutable slice for each row that's going
            // to be updated, and a regular slice for the source row.
            // We can't have both mutable and immutable borrows,
            // though, so we have to clone the source row.

            // eprintln!("index is {}, rowsize is {}, diag is {}",
            //        index, rowsize, diag);

            let mut source_row = vec![S::E::zero(); rowsize];
            eprintln!("source_row has length {}", source_row.len());
            let source_slice = &mat.as_slice()[index  .. index + rowsize];
            eprintln!("source_slice has length {}", source_slice.len());

            &source_row[..].copy_from_slice(source_slice);

            eprintln!("source_row is {:x?}", source_row);
            
            let mut mut_rows = mat.as_mut_slice().chunks_mut(cols);

            for other_row in 0..rows {
                // always consume this row
                let this_row =  mut_rows.next().unwrap();
                if other_row == diag { continue }

                // nothing to do if element already zero
                let elem : G::E = this_row[diag];
                eprintln!("Row {} has element {:x?}", other_row, elem);
                if elem == S::E::zero().into() { continue };

                // guff has a non-working "fused multiply-add"
                // function right now, so emulate it
                this_row[diag] = S::E::zero();     // save a multiply/add

                let this_row = &mut this_row[diag..]; // skip 0s
                eprintln!("this_row is {:x?}", this_row);
                for col in 0 .. rowsize {
                    eprintln!("Calculating {:x?} + {:x?} x {:x?}",
                              this_row[col],
                              elem,
                              source_row[col], );
                    this_row[col]
                        = field.add(this_row[col],
                                    field.mul(source_row[col], elem));
                    eprintln!("= {:x?}", this_row[col]);
                }
            }

            eprintln!("matrix after adding row {}: {:x?}",
                      diag,
                      // &mat.as_slice()[diag * cols .. diag * cols + cols]
                      &mat.as_slice()[ .. ]
            );

            // as we move down the diagonal, each row has fewer
            // columns to process
            rowsize -= 1;
            index += cols + 1;
        }

        eprintln!("adjoined matrix after inverse: {:x?}",
                  mat.as_slice());

        // read off the adjoined data, which now has the inverse
        // matrix
        let mut output = Self::new(self.rows(), self.cols(), true);
        let mut_chunks = output.as_mut_slice().chunks_mut(self.cols());
        let mut from_chunks = mat.as_slice().chunks(self.cols());
        for chunk in mut_chunks {
            let _ = from_chunks.next();
            chunk.copy_from_slice(from_chunks.next().unwrap());
        }

        Some(output)
    }
}

pub unsafe fn simd_warm_multiply<S : Simd<E=G::E> + Copy, G>(
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
    let mut sum_vector = zero;
    let simd_width = S::SIMD_BYTES;

    // Code for read_next() that was handled in SimdMatrix has now
    // moved to Simd. We need to track those variables here.
    let mut xform_mod_index = 0;
    let mut xform_array_index = 0;
    let     xform_array = xform.as_slice();
    let     xform_size  = xform.size();
    let mut xform_ra_size = 0;
    let mut xform_ra = zero;
    

    let mut input_mod_index = 0;
    let mut input_array_index = 0;
    let     input_array = input.as_slice();
    let     input_size  = input.size();
    let mut input_ra_size = 0;
    let mut input_ra = zero;

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
    let target = n * c;         // number of dot products

    // Be agnostic about layout
    let right = if output.is_rowwise() { 1 } else { n };
    let down  = if output.is_rowwise() { c } else { 1 };

    while total_dps < target {

        // at top of loop we should always have m0, m1 full

        // apportion parts of m0,m1 to sum

        // Change: instead of doing sum-across every time we read in a
        // full simd, xor the vector with the new values

        // handle case where k >= simd_width
        while dp_counter + simd_width <= k {
            // let (part, new_m)
            // = S::sum_across_n(m0,m1,simd_width,offset_mod_simd);
            // sum = S::add_elements(sum,part);
            let (part, new_m)
                = S::extract_elements(m0,m1,simd_width,offset_mod_simd);
            sum_vector = S::sum_vectors(&sum_vector,&part);
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
        if dp_counter < k {            // If not, ...
            let want = k - dp_counter; // always strictly positive

            // eprintln!("Calling sum_across_n with m0 {:?}, m1 {:?}, n {}, offset {}",
            //      m0.vec, m1.vec, want, offset_mod_simd);
            //            let (part, new_m) = S::sum_across_n(m0,m1,want,offset_mod_simd);
            let (part, new_m) = S::extract_elements(m0,m1,want,offset_mod_simd);

            // eprintln!("got sum {}, new m {:?}", part, new_m.vec);

            sum_vector = S::sum_vectors(&sum_vector,&part);
            if offset_mod_simd + want >= simd_width {
                // consumed m0 and maybe some of m1 too
                m0 = new_m;     // nothing left in old m0, so m0 <- m1
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

        let sum = S::sum_across_simd(sum_vector);

        // handle writing and incrementing or, oc
        // let write_index = output.rowcol_to_index(or,oc);
        let write_index = or * down + oc * right;
        output.indexed_write(write_index,sum);
        or = if or + 1 < orows { or + 1 } else { 0 };
        oc = if oc + 1 < ocols { oc + 1 } else { 0 };

        sum_vector = zero;
        dp_counter = 0;
        total_dps += 1;
    }
}

// new version of above passes in a mask
//
// Unless we're dealing with really small matrices, the most common
// path through the code will be where a SIMD read does not require
// wrap-around. In that case, the call to combine_bytes (x86) or
// extract_from_offset (Arm) can reuse the same "mask" variable. Only
// when ra_size changes will a new mask have to be loaded/calculated.
pub unsafe fn new_simd_warm_multiply<S : Simd<E=G::E> + Copy, G>(
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


/// Reference matrix multiply. Doesn't use SIMD at all, but uses
/// generic Simd types to be compatible with actual Simd
/// implementations. Note that this multiply routine does not check
/// the gcd condition so it can be used to multiply matrices of
/// arbitrary sizes.
pub fn reference_matrix_multiply<S : Simd<E=G::E> + Copy, G>(
    xform  : &mut impl SimdMatrix<S, G>,
    input  : &mut impl SimdMatrix<S, G>,
    output : &mut impl SimdMatrix<S, G>,
    field  : &G)
where G : GaloisField,
<S as Simd>::E: From<<G as GaloisField>::E> + Copy + Zero + One,
<G as GaloisField>::E: From<<S as Simd>::E> + Copy + Zero + One
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

    // Be agnostic about layout
    let right = if xform.is_rowwise() { 1 } else { n };
    let down  = if input.is_rowwise() { c } else { 1 };

    for row in 0..n {
        for col in 0..c {
            let mut xform_index  = xform.rowcol_to_index(row,0);
            let mut input_index  = input.rowcol_to_index(0,col);
            let output_index = output.rowcol_to_index(row,col);

            let mut dp = S::zero_element();
            for _ in 0..k {
                dp = S::add_elements(dp, field
                                     .mul(xform_array[xform_index].into(),
                                          input_array[input_index].into()
                                     ).into());
                xform_index += right;
                input_index += down;
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


// Other typical matrix stuff:
//
// * inversion
// * adjoining (helps with inversion/solving)
// * solver
// * row/column operations (add scaled row to another)
// * transposition
// * interleaved load/store
// * copy




#[cfg(test)]
mod tests {

    use super::*;
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
    //  new_xform_reader!(the_struct, 3, 4, 1, r0, r1);
    //  assert_eq!(the_struct.k, 3);
    //  assert_eq!(the_struct.n, 4);
    //  assert_eq!(the_struct.w, 1);
    //  the_struct.xptr += 1;
    //  assert_eq!(the_struct.xptr, 1);
    // }

    #[test]
    // test taken from simulator.rs
    #[cfg(any(target_arch = "x86", target_arch = "x86_64",
              all(any(target_arch = "aarch64", target_arch = "arm"),
                  feature = "arm_vmull")))]
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
            let mut transform = // mut because of iterator
                Matrix::new(9,9,true);
            transform.fill(&identity[..]);

            // 17 is coprime to 9
            let mut input =
                Matrix::new(9,17,false);
            let vec : Vec<u8> = (1u8..=9 * 17).collect();
            input.fill(&vec[..]);
            
            let mut output =
                Matrix::new(9,17,false);

            // works if output is stored in colwise format
            simd_warm_multiply(&mut transform, &mut input, &mut output);
            // array has padding, so don't compare that
            assert_eq!(output.array[0..9*17], vec);
        }
    }

    #[test]
    // test taken from simulator.rs
    #[cfg(any(target_arch = "x86", target_arch = "x86_64",
              all(any(target_arch = "aarch64", target_arch = "arm"),
                  feature = "arm_vmull")))]
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
            let mut transform = // mut because of iterator
                Matrix::new(18,9,true);
            transform.fill(&double_identity[..]);

            // 17 is coprime to 9
            let mut input =
                Matrix::new(9,17,false);
            let vec : Vec<u8> = (1u8..=9 * 17).collect();
            input.fill(&vec[..]);

            let mut output =
                Matrix::new(18,17,true);

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
            //  eprintln!("chunk {} has size {}", which, chunk.len());
            //  assert_ne!(which, 2); // enumerate only 0, 1
            //  if which == 1 { assert_eq!(chunk, vec)};
            // }
        }
    }

    // test conformance with a variety of matrix sizes
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64",
              all(any(target_arch = "aarch64", target_arch = "arm"),
                  feature = "arm_vmull")))]
    fn test_ref_simd_conformance() {
        let cols = 19;
        for k in 4..9 {
            for n in 4..17 {
                eprintln!("testing n={}, k={}", n, k);
                unsafe {
                    let mut transform = // mut because of iterator
                        Matrix
                        ::new(n,k,true);
                    let mut input =
                        Matrix
                        ::new(k,cols,false);

                    transform.fill(&(1u8..).take(n*k).collect::<Vec<u8>>()[..]);
                    input.fill(&(1u8..).take(k*cols).collect::<Vec<u8>>()[..]);

                    let mut ref_output = Matrix::new(n,cols,true);

                    let mut simd_output = Matrix::new(n,cols,true);

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

    #[test]
    fn test_inverse() {
        let xform = vec![
            0x35, 0x36, 0x82, 0x7a ,0xd2, 0x7d, 0x75, 0x31,
            0x0e, 0x76, 0xc3, 0xb0, 0x97, 0xa8, 0x47, 0x14,
            0xf4, 0x42, 0xa2, 0x7e, 0x1c, 0x4a, 0xc6, 0x99,
            0x3d, 0xc6, 0x1a, 0x05, 0x30, 0xb6, 0x42, 0x0f,
            0x81, 0x6e, 0xf2, 0x72, 0x4e, 0xbc, 0x38, 0x8d,
            0x5c, 0xe5, 0x5f, 0xa5, 0xe4, 0x32, 0xf8, 0x44,
            0x89, 0x28, 0x94, 0x3c, 0x4f, 0xec, 0xaa, 0xd6,
            0x54, 0x4b, 0x29, 0xb8, 0xd5, 0xa4, 0x0b, 0x2c,
        ];
        let inverse = vec![
            0x3e, 0x02, 0x23, 0x87, 0x8c, 0xc0, 0x4c, 0x79,
            0x5d, 0x2b, 0x2a, 0x5b, 0x7e, 0xfe, 0x25, 0x36,
            0xf2, 0xa9, 0xb5, 0x57, 0xa2, 0xf6, 0xa2, 0x7d,
            0x11, 0x5e, 0xe4, 0x61, 0x59, 0xf4, 0xb9, 0x42,
            0xd5, 0x16, 0xb8, 0x5b, 0x30, 0x85, 0x1e, 0x72,
            0x3b, 0xf7, 0x1b, 0x5b, 0x4c, 0x55, 0x35, 0x04,
            0x58, 0x95, 0x73, 0x33, 0x8a, 0x77, 0x1c, 0xf4,
            0x59, 0xc0, 0x7b, 0x13, 0x9f, 0x8b, 0xbe, 0xe3,
        ];

        let mut xmat = Matrix::new(8,8,true);
        xmat.fill(xform.as_slice());

        let mut out = xmat.invert(&new_gf8(0x11b,0x1b)).unwrap();

        assert_eq!(format!("{:2x?}", out.as_slice()),
                   format!("{:2x?}", inverse.as_slice()));


        // test whether invert(invert(matrix)) = identity
        let identity = Matrix::identity(8,true);

        let mut ref_output = Matrix::new(8,8,true);
        reference_matrix_multiply(&mut xmat,
                                  &mut out,
                                  &mut ref_output,
                                  &new_gf8(0x11b, 0x1b));

        assert_eq!(format!("{:2x?}", ref_output.as_slice()),
                   format!("{:2x?}", identity.as_slice()));
    }

    #[test]
    fn test_2x2_inverse() {

        // calculate inverse by hand:
        // for [ a, b, c, d ], inverse is:
        //
        //    1     |  d -b |
        // -------  | -c  a |
        // ad - bc

        let f = new_gf8(0x11b, 0x1b);
        let a = 1; let b = 2; let c = 3; let d = 4;

        let xform = vec![ a, b, c, d ];

        let det = f.inv(f.mul(a,d) ^ f.mul(b,c));

        let inverse = vec![ f.mul(det, d), f.mul(det, b),
                           f.mul(det, c), f.mul(det, a)];

        let mut xmat = Matrix::new(2,2,true);
        xmat.fill(xform.as_slice());

        let mut out = xmat.invert(&f).unwrap();

        assert_eq!(format!("{:2x?}", out.as_slice()),
                   format!("{:2x?}", inverse.as_slice()));


        // test whether invert(invert(matrix)) = identity
        let identity = Matrix::identity(2,true);

        let mut ref_output = Matrix::new(2,2,true);
        reference_matrix_multiply(
                                  &mut xmat,
            &mut out,
                                  &mut ref_output,
                                  &new_gf8(0x11b, 0x1b));

        assert_eq!(format!("{:2x?}", ref_output.as_slice()),
                   format!("{:2x?}", identity.as_slice()));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64",
              all(any(target_arch = "aarch64", target_arch = "arm"),
                  feature = "arm_vmull")))]
    #[test]
    fn test_new_simd_conformance() {
        let cols = 19;
        for k in 4..9 {
            for n in 4..17 {
                eprintln!("testing n={}, k={}", n, k);
                unsafe {
                    let mut transform = Matrix::new(n,k,true);
                    let mut input = Matrix::new(k,cols,false);

                    transform.fill(&(1u8..).take(n*k).collect::<Vec<u8>>()[..]);
                    input.fill(&(1u8..).take(k*cols).collect::<Vec<u8>>()[..]);

                    let mut new_output = Matrix::new(n,cols,true);
                    let mut old_output = Matrix::new(n,cols,true);

                    // do multiply both ways
                    simd_warm_multiply(&mut transform, &mut input,
                                       &mut old_output);
                    new_simd_warm_multiply(&mut transform, &mut input,
                                       &mut new_output);

                    assert_eq!(format!("{:x?}", old_output.as_slice()),
                               format!("{:x?}", new_output.as_slice()));
                }
            }
        }
    }


}
