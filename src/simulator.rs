//! Non-SIMD, pure Rust simulation of matrix multiplication algorithm
//!
//! See [warm_multiply] or program source for details.

// Wrap-around reads/multiplies on matrices
//
// Basics of how to do this with SIMD reads:
//
// * all reads will be aligned (assuming start of matrix is)
// * use read buffers comprising two SIMD registers 
// * extract a full SIMD register from buffer
// * as data is extracted, set reg0 = reg1, reg1 = 0
// * (this is passed on to the multiply engine)
//
// If the matrix is a multiple of the SIMD register, there's no
// challenge at all:
//
// * reads cannot go past end of matrix
// * looping just consists of resetting the read pointer
//
// At end of matrix buffer:
//
//         r0               r1
// +----------------+----------------+
// |   :            |   :            |
// +----------------+----------------+
//     `................'
//
// 
// The dotted area represents the last part of the matrix. After it,
// r1 should start reading from the start of the matrix again. If we
// use an aligned read, we will have to combine the part of r1 that we
// already have with a shifted version
// 
//         r0               r1               r2
// +----------------+----------------+----------------+
// |   :            |   :            |   :            |
// +----------------+----------------+----------------+
//     `................'................'
//         end matrix     restart matrix 
//
// (r2 won't be explicitly stored, but it helps to show the problem)
//
// At the 'restart matrix' step, we are reading from aligned memory,
// but we have to spread it over two registers, so we will have a mask
// (to select the part of r1 that we got in the last loop) combined
// with a shift.
//
// In terms of masks:
//
// r1 <- (r1 & old_bytes) | (new_bytes >> overlap)
// r2 <- (new_bytes << (16 - overlap)
//
// If we can ensure that the empty bytes are zero, it just becomes a
// problem involving 'or' and a couple of shifts. The amount of the
// shift can be stored in a normal (non-vector) register, and regular
// logical left and right shifts ensure that the newly added bits are
// always zero.
//
// The only downside is that all memory reads involve a couple of
// extra shifts, which may not be necessary in all cases (ie, where
// the matrix is a multiple of the simd width).
//
// The only subtlety involved here is handling the overlapping
// reads. For example, if we have r0 already, we have to detect that
// reading r1 involves an overlap, and that requires reading overlap
// bytes from the end of the matrix and then doing a second read from
// the start of the matrix and correctly or'ing them together, plus
// correctly calculating the new overlap value (which will be applied
// to all subsequent reads). Again, this can be done by basic
// arithmetic (if read_would_overlap {...}), and it doesn't influence
// our choice of SIMD intrinsics.
//
// Note that if we zero out (simd - 1) elements directly after the
// matrix as it is stored in memory, we can later OR it with the
// correct data taken from the (shifted) restart. Also, it prevents us
// from reading from uninitialised memory.
//
// This is nice and simple and easily portable without needing to
// delve too deeply into the docs.


// Apportionment of subproducts to output dot products
//
// Every kw bytes that we process from the input generates enough
// multiplications 
//
// kw might be > simd size, in which case we have three subcases:
//
// a) the product vector lies entirely within this kw range
// b) the start of the vector belongs to the previous range
// c) the end of the vector belongs to the next range
//
// In the case of kw <= simd size, the range can appear at the start,
// the end, the middle, or it can straddle either end.
//
// It might be possible to use masks, but we would have to use
// register pairs since we can't just rotate the mask without them.
//
//         r0               r1               r2
// +----------------+----------------+----------------+
// |   :          : |         :      |    :           |
// +----------------+----------------+----------------+
//     `..........'...........'...........'
//       this kw     next kw      ...
//
//
// For kw <= simd, the mask is the same width all the time and we
// rotate it to the right by kw each time. (and shift left by simd
// every time we consume r0)
//
// For kw > simd, we have two masks, one for the start of the range
// and one for the end. We rotate both of them every time we consume
// kw products:
//
//         r0               r1               r2
// +----------------+----------------+----------------+
// |   :            |                |    :           |
// +----------------+----------------+----------------+
//     `..................................'
//       start mask +  full vector   + end mask
//
// Masks need not be explictly stored. We can copy the vector and use
// a pair of shifts to mask out only the portion we're interested in.
//
// Note that there is no need to keep the full kw range in a set of
// registers. Instead, we can calculate the sum for each simd (or
// sub-simd) region and accumulate it in a single u8.
// 
// Sum across dot product
//
// When summing across a full 16-bit vector, we can do this with 4
// rotates and 4 xors:
//
// v ^= rot(v, 8)
// v ^= rot(v, 4)
// v ^= rot(v, 2)
// v ^= rot(v, 1)
//
// All elements of the vector will then contain the sum.
//
// The order of the operations does not matter, so if we wanted to
// only do as many shifts as needed, we could loop, starting with the
// smallest rotation and working up to the largest. Something like:
//
// next_power_of_2 = 1
// while remaining < next_power_of_2
//   v ^= rot(v, next_power_of_2)
//   next_power_of_2 <<= 1
//
// (the sum will not be spread across all elements of the vector,
// though. I think it comes out in element next_power_of_2 - 1)
//
//
// Output tape
//
// After calculating a dot product, we store it at the current output
// pointer, then we advance along the diagonal. When we go past the
// bottom or right side of the matrix, we reset the row or column,
// respectively, to zero.
//
// If c has a factor that is coprime to k and n, then each time we
// wrap around the output matrix, we will be starting from a new point
// in the first column so that after n wrap-arounds we will be
// starting again at 0,0 and the whole output matrix will have been
// filled.
//
// Open Question
//
// My original implementation used only a single register for storing
// products. It also kept 'total' and 'next_total' as complete
// vectors. It used masks to:
//
// * extract parts at the start of the current vector as belonging to
//   the current dot product
//
// * the inverse mask to apportion the remaining 
//
// Some questions:
//
// * whether explicit masks are needed or appropriate (would shifts be
//   better?)
//
// * whether to implement the product tape as two registers or stick
//   with the one (considering that we end up with two registers
//   anyway, for 
//
// * whether my original code was correct for both the kw <= simd and
//   kw >simd cases
//
// One important feature of the mask is that it zeroes out products
// from the previous dot product.
//
//


// Simulation
//
//
// just set w=1 and type = u8 for convenience

use crate::*;
use core::mem::size_of;

#[derive(Debug, Clone)]
pub struct TransformMatrix {
    n : usize,			// rows
    k : usize,			// cols
    array : Vec<u8>,		// colwise storage
    read_pointer : usize,
}

impl TransformMatrix {
    fn new(n : usize, k : usize) -> Self {
	let array = vec![0; n * k];
	let read_pointer = 0;
	Self { n, k, array, read_pointer }
    }
    fn fill(&mut self, slice : &[u8]) -> &Self {
	self.array.copy_from_slice(slice);
	self
    }
}

// infinite tape... we'll use take(simd_width)
impl Iterator for TransformMatrix {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
	let val = self.array[self.read_pointer];
	self.read_pointer += 1;
	if self.read_pointer >= self.n * self.k {
	    self.read_pointer -= self.n * self.k;
	}
	Some(val)
    }
}

#[derive(Debug, Clone)]
pub struct InputMatrix {
    k : usize,			// rows
    c : usize,			// cols
    array : Vec<u8>,		// colwise storage
    read_pointer : usize,
}

impl InputMatrix {
    fn new(k : usize, c : usize) -> Self {
	let array = vec![0; k * c];
	let read_pointer = 0;
	Self { k, c, array, read_pointer }
    }
    fn fill(&mut self, slice : &[u8]) -> &mut Self {
	self.array.copy_from_slice(slice);
	self
    }
}

// infinite tape... we'll use take(simd_width)
impl Iterator for InputMatrix {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
	let val = self.array[self.read_pointer];
	self.read_pointer += 1;
	if self.read_pointer >= self.c * self.k {
	    self.read_pointer -= self.c * self.k;
	}
	Some(val)
    }
}

use guff::*;

// MultiplyStream will "zip" the two iters above
// #[derive(Debug)]
pub struct MultiplyStream<'a> {
    // We don't care what type is producing the u8s
    xform : &'a mut Iterator<Item=u8>,
    input : &'a mut Iterator<Item=u8>,
    // can't I store a ref to something implementing GaloisField?
    // field : &'a dyn GaloisField<E=u8, EE=u16, SEE=i16>,
    field : &'a F8,  // use concrete implementation instead
}

impl<'a> Iterator for MultiplyStream<'a> {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
	let a = self.xform.next().unwrap();
	let b = self.input.next().unwrap();

	Some(self.field.mul(a,b))
    }
}

#[derive(Debug)]
pub struct OutputMatrix {
    n : usize,			// rows
    c : usize,			// cols
    array : Vec<u8>,		// 
    row : usize,
    col : usize,
    rowwise : bool
}

impl OutputMatrix {
    fn new(n : usize, c : usize, rowwise : bool) -> Self {
	let array = vec![0; n * c];
	let row = 0;
	let col = 0;
	Self { n, c, array, row, col, rowwise }
    }
    fn new_rowwise(n : usize, c : usize) -> Self {
	Self::new(n, c, true)
    }
    fn new_colwise(n : usize, c : usize) -> Self {
	Self::new(n, c, false)
    }
    fn write_next(&mut self, e : u8) {
	let size = self.n * self.c;

	if self.rowwise {
	    self.array[self.row * self.c + self.col] = e;
	} else {
	    // if col-wise (like input matrix)
	    self.array[self.row + self.col * self.n] = e;
	}

	self.row += 1;
	if self.row == self.n { self.row = 0 }
	self.col += 1;
	if self.col == self.c { self.col = 0 }
    }
}

/// "warm": "wrap-around read matrix"
///
/// This routine treats the transform and input matrices as being
/// infinite streams, multiplies them, then apportions the products to
/// the correct dot product sums. Completed dot products are written
/// out sequentially along the diagonal of the output matrix.
///
/// Provided the number of columns in the input and output matrices
/// (both set to the same value) has a factor that is relatively prime
/// to both dimensions of the transform matrix, the diagonal traversal
/// of the output matrix is guaranteed to write to every cell in the
/// matrix.
///
/// xform and input are assumed to have data in them (call
/// `matrix.fill(...)` first).
pub fn warm_multiply(xform  : &mut TransformMatrix,
                     input  : &mut InputMatrix,
                     output : &mut OutputMatrix) {

    // using into_iter() below moves ownership, so pull out any data
    // we need first
    let c = input.c;
    let n = xform.n;
    let k = xform.k;

    assert!(k > 0);
    assert!(n > 0);
    assert!(c > 0);
    assert_eq!(input.k, k);
    assert_eq!(output.c, c);
    assert_eq!(output.n, n);

    // searching for prime factors ... needs more work
    assert_ne!(k, gcd(k,c));
    
    // set up a MultiplyStream
    let xiter = xform.into_iter();
    let iiter = input.into_iter();

    let mut mstream = MultiplyStream {
        xform : xiter,
        input : iiter,
        field : &guff::new_gf8(0x11b,0x1b),
    };

    // the algorithm is trivial once we have an infinite stream
    let mut dp_counter  = 0;
    let mut partial_sum = 0u8;

    // multiplying an n * k transform by a k * c input matrix:
    //
    // n * c dot products per matrix multiplication
    //
    // k multiplies per dot product
    //
    // Grand total of n * k * c multiplies:
    let mut m = mstream.take(n * k * c);
    loop {
	// actual SIMD code will get 8 or 16 values at a time, but for
	// testing the algorithm, it's OK to go byte-by-byte
        let p = m.next();
        if p == None { break }

        let p = p.unwrap();
        // eprintln!("Product: {}", p);

        // add product to sum
        partial_sum ^= p;
        dp_counter += 1;

        // dot-product wrap around
        if dp_counter == k {
            output.write_next(partial_sum);
            partial_sum = 0u8;
            dp_counter = 0;
        }
    }
}

// Interleaving
//
// The fast matrix multiply works best when all reads (apart from
// wrap-around reads) are contiguous in memory. That suits the case
// where we're encoding using RS, striping or IDA, since each column
// of the input message corresponds to a contiguous chunk of input.
//
// When decoding, though, the input has row-wise organisation: each
// stripe, share or parity is contiguous.
//
// For this case, we need to interleave k contiguous streams.
//
// Note that there's no need to de-interleave on the output, since we
// can choose between row-wise and col-wise writes. Neither should
// have any impact on the speed of the program, since we never read
// from the output matrix.

// pass in a vector of slices, and interleave them into another slice
pub fn interleave_streams(dest : &mut [u8], slices : &Vec<&[u8]>) {

    let cols = dest.len() / slices.len();
    let mut dest = dest.iter_mut();
    let mut slice_iters : Vec::<_> = Vec::with_capacity(slices.len());
    for s in slices {
	let mut iter = s.iter();
	slice_iters.push(iter);
    }

    for _ in 0 .. cols {
	for mut slice in &mut slice_iters {
	    *dest.next().unwrap() = *slice.next().unwrap();
	}
    }
}


// SIMD simulation
//
// I've looked at various options for laying out the code in a
// portable way. I'm going to implement that here first using a
// simulated SIMD engine that works with [u8;8] as its native type.

// This trait will be in main module and will have to be implemented
// for each architecture
pub trait Simd {
    type E;			// elemental type, eg u8
    type V;			// vector type, eg [u8; 8]

    // cross product will be our simd multiply in real arch
    fn cross_product(a : Self, b : Self) -> Self;

    // to-do: consume and sum products
    fn sum_across_n(m0 : Self, m1 : Self, n : usize, off : usize) -> (Self::E, Self);
}

// Newtype for our fake architecture

// It's usual to implement, eg Simd(u128), but then we don't know what
// type the lanes are. We could do a variant, eg SimdU8(u128) to
// indicate that there are 16 u8s in the vector, but I'll use
// associated types (see Simd, above).

#[derive(Debug, Clone, Copy)]
pub struct SimSimd {
    vec : [u8; 8],
}

impl Simd for SimSimd {
    type V = [u8; 8];
    type E = u8;

    fn cross_product(a : Self, b : Self) -> Self {
	let mut prod = [0u8; 8];
	let f = new_gf8(0x11b,0x1b);
	for i in 0..8 {
	    prod[i] = f.mul(a.vec[i], b.vec[i])
	}
	Self { vec : prod }
    }
    // 
    fn sum_across_n(m0 : Self, m1 : Self, mut n : usize, off : usize) -> (Self::E, Self) {
	assert!(n <= 8);
	let mut sum = 0u8;
	if off + n >= 8 {	// straddle, will return m1
	    // let next_n = n + off - 8;
	    for i in off .. 8 { sum ^= m0.vec[i] }
	    n -= 8 - off;	// can become zero
	    for i in   0 .. n {  sum ^= m1.vec[i] }
	    // we don't change m1, but some routines might
	    return (sum, m1)
	} else {		// non-straddling, will return m0
	    for i in off .. off + n { sum ^= m0.vec[i] }
	    return (sum, m0)
	}
    }
}

// Can I reuse the above matrices? Yes, I think so. I'll use the same
// read_pointer variable that's already in them.
//
// Actually no. Not like this, anyway (conflicting implementations):
// impl Iterator for InputMatrix {
//     type = SimSimd;
//     fn next(&mut self) -> Option<Self::Item> {
// 	let val = self.array[self.read_pointer];
// 	Some(val)
//     }
// }
//
// There may be another way. It involves wrapping the original...
// struct SimSimdInputMatrix(InputMatrix);
// impl SimSimdInputMatrix {
//     fn new(k : usize, c : usize) -> Self {
// 	Self(InputMatrix::new(k,c))
//     }
//     fn fill(&mut self, slice : &[u8]) -> &mut Self {
// 	&mut Self(*self.0.fill(slice))
//     }
// }
//
// You know what? That's more effort than just copying the original.
//
// I guess that I could make a newtype over just the array's contents
// (a slice) and then have a trait SimSimdSpecificIterator local to
// this module. It's the same as the wrapping above, but with no
// pass-through required, I think. The downside is that we lose out on
// being able to use the generic matrix multiply code.
//
// Anyhow, since it's only a small bit of repeated code, and different
// SIMD iterators will need different setups anyway, it's not worth
// worrying about.

#[derive(Debug)]
pub struct SimSimdInputMatrix {
    k : usize,			// rows
    c : usize,			// cols
    array : Vec<u8>,		// colwise storage
    read_pointer : usize,	// other simd engines may differ here
}

impl SimSimdInputMatrix {
    // rather than copying this code, could just use
    // SimSimdInputMatrix { k: ...  }-style construction (cuts down on
    // boilerplate here, but more of it in test cases/demo code.
    fn new(k : usize, c : usize) -> Self {
	let array = vec![0; k * c];
	let read_pointer = 0;
	Self { k, c, array, read_pointer }
    }
    fn fill(&mut self, slice : &[u8]) -> &mut Self {
	self.array.copy_from_slice(slice);
	self
    }
}

impl Iterator for SimSimdInputMatrix {
    type Item = SimSimd;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
	// let simd_width = size_of::<SimSimd>();
 	let mut val = [0u8;8];
	let mut offset = self.read_pointer;
	for i in 0..8 {
	    val[i] = self.array[offset];
	    offset += 1;
	    if offset == self.k * self.c {
		offset = 0;
	    }
	}
	self.read_pointer = offset;
 	Some(SimSimd{vec : val})
    }
}

// same for TransformMatrix, but skip constructor boilerplate. Prefix
// all names with SimSimd as before:
#[derive(Debug, Clone)]
pub struct SimSimdTransformMatrix {
    n : usize,			// rows
    k : usize,			// cols
    array : Vec<u8>,		// colwise storage
    read_pointer : usize,
}

// put back in "boilerplate"
impl SimSimdTransformMatrix {
    fn new(n : usize, k : usize) -> Self {
	let array = vec![0; n * k];
	let read_pointer = 0;
	Self { n, k, array, read_pointer }
    }
    fn fill(&mut self, slice : &[u8]) -> &Self {
	self.array.copy_from_slice(slice);
	self
    }
}

// implementation differences: (rowwise) layout and naming of
// rows/cols.
//
// I would have saved some typing if I had just made a "Matrix" and
// stored rowwise/colwise parameter)
impl Iterator for SimSimdTransformMatrix {
    type Item = SimSimd;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
	// let simd_width = size_of::<SimSimd>();
 	let mut val = [0u8;8];
	let mut offset = self.read_pointer;
	for i in 0..8 {
	    val[i] = self.array[offset];
	    offset += 1;
	    if offset == self.n * self.k {
		offset = 0;
	    }
	}
	self.read_pointer = offset;
 	Some(SimSimd{vec : val})
    }
}

// No need to reimplement OutputMatrix

// Won't implement the pseudo "multiply" stream

// This routine should be a good basis for a generic routine:
//
// warm_multiply<S : Simd>(...)
//
// (at least if I've figured things out correctly)

pub fn simsimd_warm_multiply(xform  : &mut SimSimdTransformMatrix,
			     input  : &mut SimSimdInputMatrix,
			     output : &mut OutputMatrix) {

    // using into_iter() below moves ownership, so pull out any data
    // we need first
    let c = input.c;
    let n = xform.n;
    let k = xform.k;

    assert!(k > 0);
    assert!(n > 0);
    assert!(c > 0);
    assert_eq!(input.k, k);
    assert_eq!(output.c, c);
    assert_eq!(output.n, n);

    // searching for prime factors ... needs more work?
    if k != 1 { assert_ne!(k, gcd(k,c)) }
    
    // set up iterators
    let xiter = xform.into_iter();
    let iiter = input.into_iter();
    let field = guff::new_gf8(0x11b,0x1b);
    
    // algorithm not so trivial any more, but still quite simple
    let mut dp_counter  = 0;
    let mut sum         = 0u8;

    // we don't have mstream any more since we handle it ourselves

    // read ahead two products

    let mut i0 : SimSimd;
    let mut x0 : SimSimd;

    // Question: can rustc determine that None is never returned?
    x0 = xiter.next().unwrap();
    i0 = iiter.next().unwrap();
    let mut m0 = SimSimd::cross_product(x0,i0);

    x0  = xiter.next().unwrap();
    i0  = iiter.next().unwrap();
    let mut m1  = SimSimd::cross_product(x0,i0);

    let mut offset_mod_simd = 0;
    let mut total_dps = 0;
    let target = n * k * c;
    
    while total_dps < target {

	// actual SIMD code: will get 8 values at a time
	
	// at top of loop we should always have m0, m1 full

	// apportion parts of m0,m1 to sum

	// handle case where k >= simd_width
	while dp_counter + 8 <= k {
	    let (part, new_m) = SimSimd::sum_across_n(m0,m1,8,offset_mod_simd);
	    sum ^= part;
	    m0 = new_m;
	    x0  = xiter.next().unwrap();
	    i0  = iiter.next().unwrap();
	    m1  = SimSimd::cross_product(x0,i0); // new m1
	    dp_counter += 8;
	    // offset_mod_simd unchanged
	}
	// above may have set dp_counter to k already.
	if dp_counter < k {	       // If not, ...
	    let want = k - dp_counter; // always strictly positive
	    
	    // eprintln!("Calling sum_across_n with m0 {:?}, m1 {:?}, n {}, offset {}",
	    //      m0.vec, m1.vec, want, offset_mod_simd);
	    let (part, new_m) = SimSimd::sum_across_n(m0,m1,want,offset_mod_simd);

	    // eprintln!("got sum {}, new m {:?}", part, new_m.vec);

	    sum ^= part;
	    if offset_mod_simd + want >= 8 {
		// consumed m0 and maybe some of m1 too
		m0 = new_m;	// nothing left in old m0, so m0 <- m1
		x0  = xiter.next().unwrap();
		i0  = iiter.next().unwrap();
		m1  = SimSimd::cross_product(x0,i0); // new m1
	    } else {
		// got what we needed from m0 but it still has some
		// unused data left in it
		m0 = new_m;
		// no new m1
	    }
	    // offset calculation the same for both arms above
	    offset_mod_simd += want;
	    if offset_mod_simd >= 8 { offset_mod_simd -= 8 }
	}

	// sum now has a full dot product
	eprintln!("Sum: {}", sum);
        output.write_next(sum);
        sum = 0u8;
        dp_counter = 0;
	total_dps += 1;
    }
}

// Epilogue
//
// Writing the above two simulations has been very useful to me as a
// way of:
//
// * proving the logic correct (gcd property and simd product
//   apportionment)
// * implementing abstract concept of infinite tapes
// * figuring out a good division of labour and how to organise that
//   in terms of rust types
//
// From here, I should be able to quite easily implement real SIMD
// matrix multiplication based on the second simulation/prototype
// above.
//
// Observations
//
// Traits are very useful, but I think that I tend to overuse them
// when thinking about how to design something. In a couple of places,
// a more functional style would have been appropriate.
//
// Newtypes are very useful, and without them, it would be pretty
// difficult to make Simd types generic across platforms.
//
// Iterators are also a very useful feature, but I notice that in my
// code above, the compiler can't eliminate the check for a panic when
// calling unwrap(). This is despite my iterator code never returning
// None. Since I don't need any of the other features of the Iterator
// trait, I won't be using it in my "real" code.
//
// I think that I've learned a lot more about Rust thanks to this
// exercise. It's also helped me clarify some points about my original
// PS3 implementation and improve the overall design. The new Rust
// code looks a lot nicer, I think, and it shouldn't be much less
// efficient.
//
// "Real" SIMD
//
// I'll continue to use an Elem associated type for the wrapped SIMD
// vector types. I'll want to make the multiplication routine generic
// across all support SIMD architectures, including the simulated/fake
// one above.
//
// There will be a bit of extra boilerplate to let the compiler know
// that Elem is something that has xor defined for it, and that it can
// be zeroed.
//
// I'd like to have non-destructive type conversion, eg between
// Poly8x8_t (a wrapped type) and [u8;8].
//
// I'm sticking with u8 fields for the moment, but I should be able to
// implement larger fields without too much difficulty. An extra bit
// of work will be needed to deal with endian conversion there if
// input data is coming from an external source.
//
// Don't use Iterator trait.
//
// Do implement interleaver for each arch, but have default software
// (non-SIMD) implementation. Do this as a separate step when loading
// data into matrix.
//
// 

#[cfg(test)]

mod tests {

    use super::*;
    // use std::iter::Iterator;
    
    #[test]
    fn make_transform() {
	let mut input = TransformMatrix::new(4,3);
	let vec : Vec<u8> = (1u8..=12).collect();
	input.fill(&vec[..]);
	let elem = input.next();
	assert_eq!(elem, Some(1));

	// we can't use take() because it moves ownership of input, so
	// we have to call next() repeatedly.

	let mut part : Vec<u8> = Vec::with_capacity(24);
	for _ in 1..=5 { part.push(input.next().unwrap()) }

	assert_eq!(part, [2,3,4,5,6]);

	// wrapping around
	part.truncate(0);
	for _ in 1..=12 { part.push(input.next().unwrap()) }

	assert_eq!(part, [7,8,9,10,11,12,1,2,3,4,5,6]);
    }

    #[test]
    fn make_input() {
	let mut input = InputMatrix::new(4,3);
	let vec : Vec<u8> = (1u8..=12).collect();
	input.fill(&vec[..]);
	let elem = input.next();
	assert_eq!(elem, Some(1));

	// we can't use take() because it moves ownership of input, so
	// we have to call next() repeatedly.

	let mut part : Vec<u8> = Vec::with_capacity(24);
	for _ in 1..=5 { part.push(input.next().unwrap()) }

	assert_eq!(part, [2,3,4,5,6]);

	// wrapping around
	part.truncate(0);
	for _ in 1..=12 { part.push(input.next().unwrap()) }

	assert_eq!(part, [7,8,9,10,11,12,1,2,3,4,5,6]);
    }

    #[test]
    fn identity_multiply_colwise() {
	let identity = [1,0,0, 0,1,0, 0,0,1];
	let mut transform = TransformMatrix::new(3,3);
	transform.fill(&identity[..]);
	// 4 is coprime to 3
	let mut input = InputMatrix::new(3,4);
	let vec : Vec<u8> = (1u8..=12).collect();
	input.fill(&vec[..]);
	let mut output = OutputMatrix::new_colwise(3,4);

	// works if output is stored in colwise format
	warm_multiply(&mut transform, &mut input, &mut output);
	assert_eq!(output.array, vec);
    }

    #[test]
    fn identity_multiply_rowwise() {
	let identity = [1,0,0, 0,1,0, 0,0,1];
	let mut transform = TransformMatrix::new(3,3);
	transform.fill(&identity[..]);
	// 4 is coprime to 3
	let mut input = InputMatrix::new(3,4);
	let vec : Vec<u8> = (1u8..=12).collect();
	input.fill(&vec[..]);
	let mut output = OutputMatrix::new_rowwise(3,4);

	warm_multiply(&mut transform, &mut input, &mut output);

	// works only if output is stored in colwise format:
	// assert_eq!(output.array, vec);

	// need to transpose matrix (actually original list)... do it
	// by hand (actually, to be correct: interleave original)
	let mut transposed = vec![0u8; 12];
	let transposed = [ vec[0], vec[3], vec [6], vec[9],
			   vec[1], vec[4], vec [7], vec[10],
			   vec[2], vec[5], vec [8], vec[11], ];
	assert_eq!(output.array, transposed);

    }

    #[test]
    fn test_interleave() {
	let a0 = [0, 3, 6, 9];
	let a1 = [1, 4, 7, 10];
	let a2 = [2, 5, 8, 11];
	let vec = vec![&a0[..], &a1[..], &a2[..] ];

	let mut dest = vec![0 ; 12];
	interleave_streams(&mut dest, &vec);

	assert_eq!(dest, [0,1,2,3,4,5,6,7,8,9,10,11]);
    }


    // Rework above tests for simd version
    // -----------------------------------

    #[test]
    fn make_simd_transform() {
	let mut input = SimSimdTransformMatrix::new(4,3);
	let vec : Vec<u8> = (0u8..12).collect();
	input.fill(&vec[..]);

	let mut elem = input.next(); // returns a full simd vector
	assert_eq!(elem.unwrap().vec, [0u8,1,2,3,4,5,6,7]);

	// wrapping around
	elem = input.next();	// returns a full simd vector
	assert_eq!(elem.unwrap().vec, [8u8,9,10,11,0,1,2,3]);
    }

    #[test]
    fn make_simd_input() {
	let mut input = SimSimdInputMatrix::new(4,3);
	let vec : Vec<u8> = (0u8..12).collect();
	input.fill(&vec[..]);

	let mut elem = input.next(); // returns a full simd vector
	assert_eq!(elem.unwrap().vec, [0u8,1,2,3,4,5,6,7]);

	// wrapping around
	elem = input.next();	// returns a full simd vector
	assert_eq!(elem.unwrap().vec, [8u8,9,10,11,0,1,2,3]);
    }

    #[test]
    fn simd_identity_multiply_colwise() {
	let identity = [1,0,0, 0,1,0, 0,0,1];
	let mut transform = SimSimdTransformMatrix::new(3,3);
	transform.fill(&identity[..]);
	// 4 is coprime to 3
	let mut input = SimSimdInputMatrix::new(3,4);
	let vec : Vec<u8> = (1u8..=12).collect();
	input.fill(&vec[..]);
	let mut output = OutputMatrix::new_colwise(3,4);

	// works if output is stored in colwise format
	simsimd_warm_multiply(&mut transform, &mut input, &mut output);
	assert_eq!(output.array, vec);
    }

    #[test]
    fn simd_identity_multiply_rowwise() {
	let identity = [1,0,0, 0,1,0, 0,0,1];
	let mut transform = SimSimdTransformMatrix::new(3,3);
	transform.fill(&identity[..]);
	// 4 is coprime to 3
	let mut input = SimSimdInputMatrix::new(3,4);
	let vec : Vec<u8> = (1u8..=12).collect();
	input.fill(&vec[..]);
	let mut output = OutputMatrix::new_rowwise(3,4);

	simsimd_warm_multiply(&mut transform, &mut input, &mut output);

	// works only if output is stored in colwise format:
	// assert_eq!(output.array, vec);

	// need to transpose matrix (actually original list)... do it
	// by hand (actually, to be correct: interleave original)
	let mut transposed = vec![0u8; 12];
	let transposed = [ vec[0], vec[3], vec [6], vec[9],
			   vec[1], vec[4], vec [7], vec[10],
			   vec[2], vec[5], vec [8], vec[11], ];
	assert_eq!(output.array, transposed);
    }

    // Either of the above routines should have visited each code
    // pathway apart from those paths relating to k >= 8. This is due
    // to the coprime property (all possible straddling scenarios are
    // tested).

    // To test the paths relating to k >= 8, just use bigger identity
    // matrices.
    #[test]
    fn simd_identity_k8_multiply_colwise() {
	let identity = [
	    1,0,0,0 ,0,0,0,0,
	    0,1,0,0 ,0,0,0,0,
	    0,0,1,0 ,0,0,0,0,
	    0,0,0,1 ,0,0,0,0,
	    0,0,0,0 ,1,0,0,0,
	    0,0,0,0 ,0,1,0,0,
	    0,0,0,0 ,0,0,1,0,
	    0,0,0,0 ,0,0,0,1,
	];
	let mut transform = SimSimdTransformMatrix::new(8,8);
	transform.fill(&identity[..]);
	// 7 is coprime to 8
	let mut input = SimSimdInputMatrix::new(8,7);
	let vec : Vec<u8> = (1u8..=56).collect();
	input.fill(&vec[..]);
	let mut output = OutputMatrix::new_colwise(8,7);

	// works if output is stored in colwise format
	simsimd_warm_multiply(&mut transform, &mut input, &mut output);
	assert_eq!(output.array, vec);
    }

    #[test]
    fn simd_identity_k9_multiply_colwise() {
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
	let mut transform = SimSimdTransformMatrix::new(9,9);
	transform.fill(&identity[..]);
	// 17 is coprime to 9
	let mut input = SimSimdInputMatrix::new(9,17);
	let vec : Vec<u8> = (1u8..=9 * 17).collect();
	input.fill(&vec[..]);
	let mut output = OutputMatrix::new_colwise(9,17);

	// works if output is stored in colwise format
	simsimd_warm_multiply(&mut transform, &mut input, &mut output);
	assert_eq!(output.array, vec);
    }

    // Also test "degenerate" cases where matrices are less than simd
    // size. The real SIMD code might not be able to handle this
    // properly. At least not without specially-written wrap-around
    // matrix implementations.
    #[test]
    fn simd_identity_k1_multiply_colwise() {
	let identity = [ 1, ];
	let mut transform = SimSimdTransformMatrix::new(1,1);
	transform.fill(&identity[..]);
	// 2 is coprime to 1
	let mut input = SimSimdInputMatrix::new(1,2);
	let vec : Vec<u8> = (1u8..=2).collect();
	input.fill(&vec[..]);
	let mut output = OutputMatrix::new_colwise(1,2);

	// works if output is stored in colwise format
	simsimd_warm_multiply(&mut transform, &mut input, &mut output);
	assert_eq!(output.array, vec);
    }

    #[test]
    fn simd_identity_k2_multiply_colwise() {
	let identity = [
	    1,0,
	    0,1,
	];
	let mut transform = SimSimdTransformMatrix::new(2,2);
	transform.fill(&identity[..]);
	// 7 is coprime to 2
	let mut input = SimSimdInputMatrix::new(2,7);
	let vec : Vec<u8> = (1u8..=14).collect();
	input.fill(&vec[..]);
	let mut output = OutputMatrix::new_colwise(2,7);

	// works if output is stored in colwise format
	simsimd_warm_multiply(&mut transform, &mut input, &mut output);
	assert_eq!(output.array, vec);
    }
    
}
