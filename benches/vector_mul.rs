

use guff::{GaloisField, new_gf8, F8 };
use guff_matrix::*;
use guff_matrix::x86::*;

// despite docs, should have no main() here.
// #![allow(unused)]
//fn main() {

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use criterion::BenchmarkId;


fn ref_gf8_vec(size : usize) {
    let av = vec![0x53u8; size];
    let bv = vec![0xcau8; size];
    let mut rv = vec![0x00u8; size];

    let f = new_gf8(0x11b,0x1b);
    f.vec_cross_product(&mut rv[..], &av[..], &bv[..])
}
    
fn x86_gf8_vec(size : usize) {
    let av = vec![0x53u8; size];
    let bv = vec![0xcau8; size];
    let mut rv = vec![0x00u8; size];

    unsafe { vmul_p8_buffer(&mut rv[..], &av[..], &bv[..], 0x1b); }
}

// not sure if I can get accurate timings for this
fn alloc_only(size : usize) {
    let _av = vec![0x53u8; size];
    let _bv = vec![0xcau8; size];
    let mut _rv = vec![0x00u8; size];
}
    
fn bench_alloc_only(c: &mut Criterion) {
    c.bench_function("alloc", |b| b.iter(|| alloc_only(32768)));
}

fn bench_ref_gf8_vec(c: &mut Criterion) {
    c.bench_function("ref gf8", |b| b.iter(|| ref_gf8_vec(32768)));
}

fn bench_x86_gf8_vec(c: &mut Criterion) {
    c.bench_function("vec gf8", |b| b.iter(|| x86_gf8_vec(32768)));
}

// Test matrix multiplication
//
// Non-SIMD version (use same Simd Matrix types)

// Will model on simd_identity_k9_multiply_colwise() test from lib.rs
fn simd_x86_gf8_matrix_mul() {
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

fn bench_simd_x86_gf8_matrix_mul(c: &mut Criterion) {
    c.bench_function("simd gf8 matrix",
		     |b| b.iter(|| simd_x86_gf8_matrix_mul()));
}



fn ref_gf8_matrix_mul() {

    let f = new_gf8(0x11b, 0x1b);
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
	let mut xform =		// mut because of iterator
	    X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(9,9,true);
	xform.fill(&identity[..]);

	// 17 is coprime to 9
	let mut input =
	    X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(9,17,false);
	let vec : Vec<u8> = (1u8..=9 * 17).collect();
	input.fill(&vec[..]);

	// output layout does matter for final assert!()
	let mut output =
	    X86SimpleMatrix::<x86::X86u8x16Long0x11b>::new(9,17, false);

	// 
	// simd_warm_multiply(&mut transform, &mut input, &mut output);
	//

	// our implementation
	let k = 9;
	let c = 17;

	for row in 0..k {
	    for col in 0..c {
		let xform_index  = xform.rowcol_to_index(row,0);
		let input_index  = input.rowcol_to_index(0,col);
		let output_index = output.rowcol_to_index(row,col);

		let mut dp = 0;
		for i in 0..k {
		    dp ^= f
			.mul(xform.array[xform_index + i],
			     input.array[input_index + i]
			);
		}
		output.array[output_index] = dp;
	    }
	}
	
	// array has padding, so don't compare that
	assert_eq!(output.array[0..9*17], vec);
    }

}

fn bench_ref_gf8_matrix_mul(c: &mut Criterion) {
    c.bench_function("ref gf8 matrix",
		     |b| b.iter(|| ref_gf8_matrix_mul()));
}




// Use like-for-like harness (bench_with_input instead of
// bench_function)

criterion_group!(benches,
		 // 0.1.3
		 // bench_alloc_only,
		 bench_ref_gf8_vec,
		 bench_x86_gf8_vec,
		 // 0.1.5 (bench before release)
		 bench_simd_x86_gf8_matrix_mul,
		 bench_ref_gf8_matrix_mul,
);
criterion_main!(benches);

//}
// Sample output ...
// ref gf8                 time:   [277.33 us 277.43 us 277.55 us]
// vec gf8                 time:   [137.85 us 137.95 us 138.09 us]                    
//
// Not a big improvement. But turn on rust flags:
//
// RUSTFLAGS="-C target-cpu=native" cargo bench -q
//
// ref gf8                 time:   [292.16 us 292.26 us 292.42 us]                    
// vec gf8                 time:   [19.975 us 19.985 us 19.998 us]                     
//
// I think that the main culprit was vblend instruction being turned
// into a function call without the `target-cpu=native` option.
//
// Anyway, nearly 16 times as fast. As expected.
