

// Wrap-around reads/multiplies on matrices

// basics:
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
// When summing across a full 16-bit vector, we can do this with 3
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
// pointer, then we advance that by w*(k + n). This proceeds along the
// diagonal in the matrix. When the pointer goes past the end of the
// matrix (ie, ptr > n * c * k), we subtract the length of the matrix.
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

struct TransformMatrix {
    n : usize,			// rows
    k : usize,			// cols
    array : Vec<u8>,		// colwise storage
    read_pointer : usize,
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

struct InputMatrix {
    k : usize,			// rows
    c : usize,			// cols
    array : Vec<u8>,		// colwise storage
    read_pointer : usize,
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

// MultiplyStream will "zip" the two iters above
struct MultiplyStream<'a> {
    // We don't care what type is producing the u8s
    xform : &'a dyn Iterator<Item=u8>,
    input : &'a dyn Iterator<Item=u8>,
}	

