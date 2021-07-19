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

// xform0, xform1 are externally stored registers; this tracks
// variables needed to read simd bytes at a time from memory into
// those registers, and to extract a full simd bytes for passing to
// multiply routine.
struct TransformTape {
    k : usize,
    n : usize,
    w : usize,
    xptr : usize,    // (next) read pointer within matrix
}


// One approach: trait providing matrix multiply over infinite product
// stream/tape. Then implement the missing bits in each SIMD module.
//
trait StreamingMatrixMul {

    type Elem : std::ops::BitXor<Output=Self::Elem>; // eg, u8
    type SIMD;			// eg, __m128i

    // Make n, c, k and w associated constants.

    // The reason for this is that eventually I might want to have a
    // derive macro that builds (derives) a matrix solver given the
    // appropriate types/constants
    
    const N : usize;
    const C : usize;
    const K : usize;
    const W : usize;
    const SIMD_SIZE : usize;	// length of SIMD / length of Elem
    const DP_FINAL : usize;	// 

    // return zero of appropriate type
    fn zero_product(&self) -> Self::Elem;
    // sum across full SIMD vector
    fn sum_across(&self, v : Self::SIMD) -> Self::Elem;
    // sum across k % simd_size remainder
    fn sum_across_remaining(&self, v : Self::SIMD) -> Self::Elem;

    // these eventually read from xform/input matrices
    fn get_simd_products(&self) -> Self::SIMD;
    fn get_remaining_products(&self) -> Self::SIMD;

    // this eventually writes to output matrix
    fn write_next(&self, elem : Self::Elem);

    fn multiply(&self) {
	let mut product = self.zero_product();
	let mut dp_remaining = Self::K;
	let mut written : usize = 0;
	loop {
	    // if k >= simd_size, add simd_size products at a time
	    while dp_remaining >= Self::SIMD_SIZE {
		product = product ^ self.sum_across(self.get_simd_products());
		dp_remaining -= Self::SIMD_SIZE
	    }
	    // todo: what if we had (simd_size / 2)-way simd engine too?

	    // the remainder will always be k % simd_size
	    if dp_remaining != 0 {
		product = product ^ self.sum_across_remaining(self.get_remaining_products())
	    }
	    self.write_next(product);
	    written += 1;
	    if written == Self::N * Self::C { break }
	    product = self.zero_product();
	    dp_remaining = Self::K;
	}
    }
}

// The above could also be written to use methods to determine the
// constants. The types would still have to be associated types unless
// we rewrote the above method signatures to return no data apart from
// maybe a bool, eg:
//
// if self.is_kw_gt_simd_size() { self.add_full_simd_products() }
// if self.is_kw_mod_simd_size_ne_zero() { self.add_remainder() }
//

// What will implement this? A multiply stream.

// That, in turn, will have to:
//
// Store a transform matrix that implements wrap-around read trait
// Store an input matrix that also implements wrap-around read
// Store an output matrix that implements diagonal write
// Interact with a SIMD engine
//
// I think that I'll follow much the same idea as in the simulator
// except that instead of infinite iterator for reads, we work with
// SIMD-sized chunks.

pub trait WarmSimd {

    type Elem;
    type SIMD;

    fn read_next_simd(&self) -> Self::SIMD;
}

// Optimising for special cases:
//
// * xform matrix fits into a small number of SIMD registers
// * xform is a multiple of SIMD size
//
// In the first case, we can read the matrix into a register or
// register pair and only use rotates.
//
// In the second case, we can have simplified wrap-around code because
// there will be no need for any shifts/rotates.
//
// The logical conclusion of this sort of optimisation might be to
// work on smaller submatrices. 

// Example of first optimisation
struct SubSimdMatrixStreamer {
    buffer : [u8;8],

    // "rotate" as a name isn't quite right, but general idea is that
    // if we have at least one copy of the matrix in the register we
    // can use shifts and ors to advance the "read pointer" by an
    // arbitrary amount. For example, if simd size is 8, and matrix is
    // 6, we start with the following indexes into the matrix:
    //
    // [0, 1, 2, 3, 4, 5, 0, 1 ]
    //
    // advancing by 8, we want to start at 2 and get:
    //
    // [2, 3, 4, 5, 0, 1, 2, 3 ]
    //
    // We do so by a combination of shl and shr:
    //
    // (buffer << 2) | (buffer >> 4)
    
    shl : usize,
    shr : usize,
}

impl SubSimdMatrixStreamer {

    // constructor will read from memory and write at least one full
    // copy and possibly a partial copy into buffer

    //    fn new() -> Self { }
}

// Now we can make this struct work as a WarmSimd:
impl WarmSimd for SubSimdMatrixStreamer {
    type Elem = u8;
    type SIMD = [u8;8];		// in reality, needs to be generic
    fn read_next_simd(&self) -> Self::SIMD {
	let result = self.buffer;
	// following won't compile yet
	//	self.buffer = (result << self.shl) | (result >> self.shr);
	result
    }

}

// We can have other structures that also implement WarmSimd:
//
// * variants of SubSimdMatrixStreamer, where the array fits into two,
//   three or more SIMD registers (I'm assuming that even though these
//   values are stored in a struct, when they come to be used by
//   StreamingMatrixMul, if we inline all the functions, they'll get
//   loaded once into registers and not be written back out until the
//   multiply is done; not 100% sure that this is so, though)
//
// * more normal case where matrix does not fit in registers, so does
//   actual wrap-around read on memory.
//


// Another approach ...
//
// More of a bottom-up approach: Make some new types representing
// generic 128-bit and 64-bit SIMD types.
//
// Don't have to support every single type of operation... just enough
// to implement buffering and wrap-around reads.
//

// Rethink
//
// I will have some sort of translation/compatibility layer for
// dealing with different arches. Whether that's a wrapper around SIMD
// types or a completely different (marker) type, I'm not sure yet.
//
// I no longer think that it's a good idea to wrap up the
// StreamingMatrixMul code as a trait. Long story short, it's much
// better to use functional style. See warm_simd_multiply() below
// the supporting trait/structs:

pub trait Matrix<E> {
    const IS_ROWWISE : bool;
    fn is_rowwise(&self) -> bool { Self::IS_ROWWISE }
    fn rowcol_to_index(&self, r : usize, c : usize) -> usize {
	if Self::IS_ROWWISE {
	    r * self.rows() + c
	} else {
	    r + c * self.cols()
	}
    }
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
}

// making rowwise, colwise matrices distinct types means that any
// conversion between row,col to interior vector index doesn't have to
// test a variable is_rowwise() each time. We have a little bit of
// duplicated code/boilerplate, but not much.
pub struct RowwiseMatrix<E> {
    rows : usize,
    cols : usize,
    vec  : Vec<E>,
}
impl<E> Matrix<E> for RowwiseMatrix<E> {
    const IS_ROWWISE : bool = true;
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
}

pub struct ColwiseMatrix<E> {
    rows : usize,
    cols : usize,
    vec  : Vec<E>,
}
impl<E> Matrix<E> for ColwiseMatrix<E> {
    const IS_ROWWISE : bool = true;
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
}

// matrix structs above will also have to implement Warm_Simd<E,S>
pub trait WarmSimdMatrix<E,S> : Matrix<E> + WarmSimd<Elem=E, SIMD = S> {
    // todo
//    type Reg;
}

// 
pub fn warm_simd_multiply<E,S>(xform : &impl WarmSimdMatrix<E,S>,
			       input : &impl WarmSimdMatrix<E,S>,
			       mut output : &impl Matrix<E>) {

    if !xform.is_rowwise() { panic!("xform must be in rowwise format") }
    if input.is_rowwise()  { panic!("input must be in colwise format") }

}

// WarmSimd will iterate over the array returning a native SIMD type
//
// I was thinking that I might need to use smaller SIMD types, but
// that need not be exposed. For example, if reading from a matrix
// that doesn't have any padding at the end, we can't use full SIMD
// read, so we need to use smaller sizes (u128->u64->u32->u16->u8).
//
// The SIMD newtype could be purely descriptive (eg, ArmSimd,
// FakeSimd, etc.), in which case it would have to have an associated
// type. Alternatively, it can be a wrapping type for the actual
// storage type. The latter seems better.
//
// Can we "overload" names like ...
//
// struct Simd(u128);
// struct Simd([u8;8]);
//
// No...
//
// So we'd need something like:
// struct SliceSimd([u8;8]);
// struct PrimitiveSimd(u128);

// Then we'd need to implement WarmSimd on each.
// 
// Remembering, though, that to do wrap-around reads, we want to store
// state (current read pointer, register buffer pair), it's probably
// better to have something like this:
// trait SliceSimd {
// }
// impl WarmSimd for SliceSimd {
//    type Elem = u8;
//    type SIMD = [u8;8];
//    fn read_next_simd(&self) -> Self::SIMD { [0u8;8] }
//}

// This is going in the right direction, but I've lost the linkage to
// the Matrix types above.

// Maybe the thing to do is make the matrix type implement
// IntoIterator or something of that style. Actually, Iterator is
// better because I don't want to pass ownership. See:
//
// https://stackoverflow.com/questions/34733811/what-is-the-difference-between-iter-and-into-iter
//
// I've already implemented something similar in the simulator. I
// would mainly just have to change the Item associated type so that
// it's something that holds a SIMD-like value.
//
// The only question with that, though, is that in order to keep track
// of state in SIMD chunks, we'd have to make the Matrix types generic
// over that SIMD type as well:
//
// struct RowwiseMatrix<E,S> {
//   //... normal stuff first
//   reg0 : S,
//   reg1 : S,
//   offset_mod_simd_width : usize
// }
//
// We should only need to specify Iterator<Item=S> rather than trying
// to pass in E as well. We may need to store it as an associated
// type, though? My reason for thinking this is that one of our "fake"
// Simd types might work with [u8; 8], but may need to hold single
// values temporarily in variables? Or, if we're using larger fields,
// we might need to know that otherwise we might be mixing up endian.
//
// Another thing that I'm thinking of is decompsition of large
// matrices. If we wanted OpenCL-like threads all working on the same
// matrix, we can borrow contiguous bits for reading (and have
// iterators that can read from them in parallel), but writing using
// an iterator isn't going to work. If these were all working in
// separate threads, I guess we'd need a channel for receiving
// results, which the main thread would then place into the correct
// row,col position.
//
// Anyway, maybe I don't need E.
//
// I do need to implement the newtype Simd, though. Or, rather, the
// trait that all the SIMD structs will implement...
//
// * It definitely has to be a storage type
//
// * It will have to provide type conversion to the underlying type
//   (preferably infallibly)
//
// * It may need to have some operations (like shifting, masking,
//   selection across a pair of buffers of the type)
//
// * It will almost definitely have to implement sum-across
//
// * It will be passed back from the matrix's iter().next()
//
// * It may be used to type the SIMD multiply routines (via a thin
//   type translation layer, or that could be the only interface that
//   the multiply routine will be written to support)
//
// Question: how do iterators get their initial values set?
//
// I suppose it's just done in object creation. That is, during the
// creation of the object that implements Iterator.
//
// Anyway, to get back to the point, I can choose to "hide" the actual
// low-level implementation of wrap-around read within a custom
// struct. Or I can try to use a one-size-fits-all struct that
// provides a default implementation of the idea in terms of the
// register pair and a modular counter. With that, so long as the
// particular Simd newtype implements all the required methods
// (shifts, rotates, masks, etc.), it doesn't need to do anything
// else.
//
// "Hiding" the actual implementation means leaving all implementation
// details up to new struct. All it has to do is satisfy the basic
// Iterator (and Matrix) traits.
//
// Both approachs have pros/cons. I think that I'll go with the
// latter, leaving implementation details completely up to the
// particular SIMD architecture.

// Still more ...
//
// If we're taking SIMD elements at a time, we still have to apportion
// the products to distinct dot products. So while reading from the
// two matrices and doing pair-wise multiplications of the streams is
// fine, we still need some help
//
// Division of labour:
//
// Caller (generic code):
//
// * reads from the two iterators and pseudo multiply stream iter
// * has a pair of registers for holding partial sums
// * tracks the correct offset within this register pair
// * decides whether to add a full simd's worth or fewer of products
//
// Callee (Matrix):
//
// * implements matrix and iterator, eg small matrix may implement
//   iterator as reading from repeatedly from registers.
//
// Callee (SIMD type's associated function)
//
// * cross product
// * takes a register pair and offset and returns the sum across
//   those elements, along with updated register/register pair
//
// The last thing is stateless. Both caller and callee have to agree
// on organisation of the register pair:
//
//       r0              r1
// +---+------+---+---------------+
// |XXX| next |   |   readahead   |
// +---+------+---+---------------+
//
// Might as well always have read-ahead? Probably.
//
// In the above, callee will only return the updated register r0. 
//
// The next read after that straddles the boundary:
//
//       r0              r1
// +---+------+---+--+------------+
// |XXX|XXXXXX| next |readahead   |
// +---+------+---+--+------------+
//
// So it will return an updated r1 only and the caller will move r1
// into r0 and do another readahead, so from its point of view, the
// situation will look like:
//
//       r0'             r1'
// +--+------+----+---------------+
// |XX| next |    |   readahead   |
// +--+------+----+---------------+
//
// Could also include a one-register version if the caller knows that
// the value will not straddle two registers.
//
// The state of the X bytes isn't interesting for the caller, but the
// callee might set them to zero if it leads to a more efficient
// sum-across step.



// My first macro. I think that it will be easier to write a generic
// version of the multiply routine that works across architectures if
// I can hide both register types (eg, __m128* on x86) and intrinsics.
//
// Another advantage is that I can test the macros separately.
//
// The only fly in the ointment is if I need fundamentally different
// logic to map operations onto intrinsics...

// #[macro_export]
//macro_rules! new_xform_reader {
//    ( $s:ident, $k:expr, $n:expr, $w:expr, $r0:ident, $r1:ident) => {
//	let mut $s = TransformTape { k : $k, n : $n, w : $w, xptr : 0 };
//    }
//}

// Actually, I can eliminate explicit variable names above. That would
// solve the problem of having to use different number of variables to
// achieve a certain result.

// Only pass in that are germane to the algorithm, not the
// arch-specific implementation:
//
// init_xform_stream!(xform.as_ptr())
// init_input_stream!(input.as_ptr())
// init_output_stream!(output.as_ptr())
//
// ...

// Matrix sizes
//
// We multiply xform x input giving output
//
// These values are fixed by the transform matrix:
//
// * n: number of rows in xform
// * k: number of columns in xform
// * w: number of bytes in each element
//
// We also have simd_width, which is the width of the SIMD vectors, in
// bytes.
//
// The input matrix has k rows. The output matrix has n rows.
//
// We fix the input and output matrices as having the same number of
// columns, c. It has to have a factor f that is coprime to both kw
// and n.
//
//       k     x       c      =        c
//    +-----+     +----…---+    +----…------+
//    |     |     |        |    |           |
//  n |     |   k |        |  n |           |
//    |     |     |        |    |           |
//    |     |     +----…---+    |           |
//    +-----+                   +----…------+
//
//     xform        input          output
//
// various wrap-around boundaries:
//
// simd_width... we have two full simd registers doing aligned reads,
// but we will have to extract a single simd register worth of data
// from it. We need register pairs for:
//
// * xform stream
// * input stream
//
// We don't need them for subproducts but we need to sum these, so one
// way of keeping dot product components separate is to use a similar
// register pair setup.
//
// kw ... full dot product
//
// complicated a bit because two cases:
//
// a) kw <= simd_width
// b) kw >  simd_width
//
// In the first case, we will get at least one dot product from each
// simd multiply. In the second, we have to do several simd operations
// in order to get a full dot product.
//
// we have efficient ways of summing across vectors by using shifts
// and xor, as opposed to taking n - 1 xor steps to sum n values
//
// nkw ... wrap around transform matrix
//
// if nkw is coprime to simd_width, then we would be heading in
// non-aligned read territory here. Assuming that is the case:
//
// ah, I need two registers for products. 
//
// kwc ... wrap around right of input matrix
//
// this will be coprime to n each time we wrap around, so we always
// restart at a different row.
//

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
    unsafe fn sum_across_n_old(m0 : Self, m1 : Self, n : usize, off : usize)
			   -> (Self::E, Self);

    // helper functions for working with elemental types. An
    // alternative to using num_traits.
    fn zero_element() -> Self::E;
    fn add_elements(a : Self::E, b : Self::E) -> Self::E;
}

// For Matrix trait, I'm not going to distinguish between rowwise and
// colwise variants. The iterators will just treat the data as a
// contiguous block of memory. It's only when it comes to argument
// checking (to matrix multiply) and slower get/set methods that the
// layout matters.
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

    assert!(k > 0);
    assert!(n > 0);
    assert!(c > 0);
    assert_eq!(input.rows(), k);
    assert_eq!(output.cols(), c);
    assert_eq!(output.rows(), n);

    // searching for prime factors ... needs more work?
    if k != 1 { assert_ne!(k, gcd(k,c)) }
    
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
