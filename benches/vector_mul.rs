

use guff::{GaloisField, new_gf8, F8 };
use guff_matrix::*;

// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// use guff_matrix::x86::*;

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
    
// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// use guff_matrix::x86::*;
fn simd_gf8_vec(size : usize) {
    let av = vec![0x53u8; size];
    let bv = vec![0xcau8; size];
    let mut rv = vec![0x00u8; size];

    NativeSimd::cross_product_slices(&mut rv[..], &av[..], &bv[..]);

    // unsafe { vmul_p8_buffer(&mut rv[..], &av[..], &bv[..], 0x1b); }
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
    c.bench_function("ref gf8 vec", |b| b.iter(|| ref_gf8_vec(32768)));
}

fn bench_simd_gf8_vec(c: &mut Criterion) {
    c.bench_function("simd gf8 vec", |b| b.iter(|| simd_gf8_vec(32768)));
}

// Test matrix multiplication
//
// Non-SIMD version (use same Simd Matrix types)

// Will model on simd_identity_k9_multiply_colwise() test from lib.rs
fn simd_gf8_matrix_mul(cols : usize) {
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
        let mut transform =     // mut because of iterator
            Matrix::new(9,9,true);
        transform.fill(&identity[..]);

        // 17 is coprime to 9
        let mut input =
            Matrix::new(9,cols,false);

        // create a vector with elements 1..255 repeating
        let src : Vec<u8> = (1u8..=255).collect();
        let iter = src.into_iter().cycle().take(9*cols);
        let vec : Vec<u8> = iter.collect::<Vec<_>>();
        
        // let vec : Vec<u8> = (1u8..=9 * cols).collect();
        input.fill(&vec[..]);

        let mut output =
            Matrix::new(9,cols,false);

        // works if output is stored in colwise format
        simd_warm_multiply(&mut transform, &mut input, &mut output);
        // array has padding, so don't compare that
        assert_eq!(output.array[0..9*cols], vec);
    }
}

fn new_simd_gf8_matrix_mul(cols : usize) {
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
        let mut transform =     // mut because of iterator
            Matrix::new(9,9,true);
        transform.fill(&identity[..]);

        // 17 is coprime to 9
        let mut input =
            Matrix::new(9,cols,false);

        // create a vector with elements 1..255 repeating
        let src : Vec<u8> = (1u8..=255).collect();
        let iter = src.into_iter().cycle().take(9*cols);
        let vec : Vec<u8> = iter.collect::<Vec<_>>();
        
        // let vec : Vec<u8> = (1u8..=9 * cols).collect();
        input.fill(&vec[..]);

        let mut output =
            Matrix::new(9,cols,false);

        // works if output is stored in colwise format
        new_simd_warm_multiply(&mut transform, &mut input, &mut output);
        // array has padding, so don't compare that
        assert_eq!(output.array[0..9*cols], vec);
    }
}

fn bench_simd_gf8_matrix_mul_17(c: &mut Criterion) {
    c.bench_function("simd gf8 matrix 9x17",
                     |b| b.iter(|| simd_gf8_matrix_mul(17)));
}

fn bench_simd_gf8_matrix_mul_16384(c: &mut Criterion) {
    c.bench_function("simd gf8 matrix 9x16384",
                     |b| b.iter(|| simd_gf8_matrix_mul(16384)));
}

fn bench_new_simd_gf8_matrix_mul_16384(c: &mut Criterion) {
    c.bench_function("new simd gf8 matrix 9x16384",
                     |b| b.iter(|| new_simd_gf8_matrix_mul(16384)));
}

fn ref_gf8_matrix_mul(cols : usize) {

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
        let mut xform =         // mut because of iterator
            Matrix::new(9,9,true);
        xform.fill(&identity[..]);

        // 17 is coprime to 9
        let mut input =
            Matrix::new(9,cols,false);

        // create a vector with elements 1..255 repeating
        let src : Vec<u8> = (1u8..=255).collect();
        let iter = src.into_iter().cycle().take(9*cols);
        let vec : Vec<u8> = iter.collect::<Vec<_>>();
        // eprintln!("Vector length is {}", vec.len());

//      let vec : Vec<u8> = (1u8..=9 * 17).collect();
        input.fill(&vec[..]);

        // output layout does matter for final assert!()
        let mut output = Matrix::new(9,cols, false);

        // 
        // simd_warm_multiply(&mut transform, &mut input, &mut output);
        //

        // code that was here moved into fn in main module
        
        reference_matrix_multiply(&mut xform, &mut input, &mut output, &f);
        
        // array has padding, so don't compare that
        assert_eq!(output.array[0..9*cols], vec);
    }

}

fn bench_ref_gf8_matrix_mul_17(c: &mut Criterion) {
    c.bench_function("ref gf8 matrix 9x17",
                     |b| b.iter(|| ref_gf8_matrix_mul(17)));
}

fn bench_ref_gf8_matrix_mul_16384(c: &mut Criterion) {
    c.bench_function("ref gf8 matrix 9x16384",
                     |b| b.iter(|| ref_gf8_matrix_mul(16384)));
}


// Compare output write patterns in an attempt to determine write
// cache miss penalty. There's a big drop-off in performance between
// doing simd multiplication and using it in the wrap-around read
// multiply routine. I wonder how much of that can be attributed to
// scattered writes.


fn serial_write(cols : usize) {

    let mut out = vec![200u8; 19 * cols];

    let mut row = 0;
    let mut col = 0;

    for c in 0 .. 19 * cols {
        out[row * 19 + col] = ((c * c) & 0xffusize) as u8;

        // compiler *may* be able to optimise this somewhat
        col = if col == cols { 0 } else { col + 1 };
        row = if col == 0 { row + 1 } else { row };
    }

    // The loop is apparently not optimised out, but just iterate over
    // it once to be sure
    let mut sum = 0;
    for c in 0..19*cols {
        sum ^= out[c];
    }
    // eprintln!("final sum {}", sum);
}

fn scattered_write(cols : usize) {
    let mut out = vec![200u8; 19 * cols];

    let mut row = 0;
    let mut col = 0;

    for c in 0 .. 19 * cols {
        out[row * 19 + col] = ((c * c) & 0xffusize) as u8;

        // less likely to be abl to optimise this out (row never gets
        // to 19, except in final iteration)
        col = if col == cols { 0 } else { col + 1 };
        row = if row == 19 { 0 } else { row + 1 };
    }
    let mut sum = 0;
    for c in 0..19*cols {
        sum ^= out[c];
    }
    // eprintln!("final sum {}", sum);
}

fn bench_serial_write_4k(c: &mut Criterion) {
    c.bench_function("serial write 19 * 4k",
                     |b| b.iter(|| serial_write(4096)));
}

fn bench_scattered_write_4k(c: &mut Criterion) {
    c.bench_function("scattered write 19 * 4k",
                     |b| b.iter(|| scattered_write(4096)));
}

fn bench_serial_write_16k(c: &mut Criterion) {
    c.bench_function("serial write 19 * 16k",
                     |b| b.iter(|| serial_write(16384)));
}

fn bench_scattered_write_16k(c: &mut Criterion) {
    c.bench_function("scattered write 19 * 16k",
                     |b| b.iter(|| scattered_write(16384)));
}

fn bench_serial_write_64k(c: &mut Criterion) {
    c.bench_function("serial write 19 * 64k",
                     |b| b.iter(|| serial_write(65536)));
}

fn bench_scattered_write_64k(c: &mut Criterion) {
    c.bench_function("scattered write 19 * 64k",
                     |b| b.iter(|| scattered_write(65536)));
}


criterion_group!(benches,
                 // 0.1.3
                 // bench_alloc_only,
                 bench_ref_gf8_vec,
                 bench_simd_gf8_vec,
                 // 0.1.5 (bench before release)
                 bench_ref_gf8_matrix_mul_17,
                 bench_simd_gf8_matrix_mul_17,
                 bench_ref_gf8_matrix_mul_16384,
                 bench_simd_gf8_matrix_mul_16384,
                 // 0.1.13
                 bench_new_simd_gf8_matrix_mul_16384,
                 bench_serial_write_4k,
                 bench_scattered_write_4k,
                 bench_serial_write_16k,
                 bench_scattered_write_16k,
                 bench_serial_write_64k,
                 bench_scattered_write_64k,
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
