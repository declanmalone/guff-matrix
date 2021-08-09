
use criterion::{criterion_group, criterion_main, Criterion};
// use criterion::{black_box};
// use criterion::BenchmarkId;



#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_vmull"))]
pub mod arm_ops {

    use guff::{GaloisField, new_gf8 };
    use guff_matrix::*;
    use guff_matrix::arm_vmull::*;

    use criterion::Criterion;


    pub fn ref_gf8_vec(size : usize) {
        let av = vec![0x53u8; size];
        let bv = vec![0xcau8; size];
        let mut rv = vec![0x00u8; size];

        let f = new_gf8(0x11b,0x1b);
        f.vec_cross_product(&mut rv[..], &av[..], &bv[..])
    }
    
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // use guff_matrix::x86::*;
    pub fn simd_gf8_vec(size : usize) {
        let av = vec![0x53u8; size];
        let bv = vec![0xcau8; size];
        let mut rv = vec![0x00u8; size];

        NativeSimd::cross_product_slices(&mut rv[..], &av[..], &bv[..]);

        // unsafe { vmul_p8_buffer(&mut rv[..], &av[..], &bv[..], 0x1b); }
    }

    pub fn nar_tuple(size : usize) {

        let _rows = 19;
        let _cols = size;

        // let fill_vec = Vec::with_capacity(19 * size + 8)
        let fill_list = (0u8..=255).cycle().take(19 * size + 8);
        let mut fill_vec : Vec<u8> = fill_list.collect();

        // copy elements from 0..8 to end..end + 8
        for index in 0..8 {
            fill_vec[19 * size + index] = fill_vec[index]
        }

        // call the function
        let mut index = 0;
        let slice = &fill_vec[0..19*size+8];
        for _times in 0..size {
            let (new_index, _res) = nar_read_next_tuple(index, 19*size, slice);
            index = new_index;
        }
    }

    pub fn bench_nar_tuple_4k(c: &mut Criterion) {
        c.bench_function("nar (functional) 4k",
                         |b| b.iter(|| nar_tuple(4096)));
    }
    
    pub fn bench_nar_tuple_16k(c: &mut Criterion) {
        c.bench_function("nar (functional) 16k",
                         |b| b.iter(|| nar_tuple(16384)));
    }
    
    pub fn bench_nar_tuple_64k(c: &mut Criterion) {
        c.bench_function("nar (functional) 64k",
                         |b| b.iter(|| nar_tuple(65536)));
    }
    
    pub fn nar_mut(size : usize) {

        let rows = 19;
        let cols = size;

        // let fill_vec = Vec::with_capacity(19 * size + 8)
        let fill_list = (0u8..=255).cycle().take(rows * cols + 8);
        let mut fill_vec : Vec<u8> = fill_list.collect();

        // copy elements from 0..8 to end..end + 8
        for index in 0..8 {
            fill_vec[rows * cols + index] = fill_vec[index]
        }

        // call the function
        let mut index = 0;
        let slice = &fill_vec[0..rows * cols + 8];
        for _times in 0..size {
            let _res = nar_read_next_mut(&mut index, rows * cols, slice);
        }
    }
    
    pub fn bench_nar_mut_4k(c: &mut Criterion) {
        c.bench_function("nar (mutable) 4k",
                         |b| b.iter(|| nar_mut(4096)));
    }

    pub fn bench_nar_mut_16k(c: &mut Criterion) {
        c.bench_function("nar (mutable) 16k",
                         |b| b.iter(|| nar_mut(16384)));
    }

    pub fn bench_nar_mut_64k(c: &mut Criterion) {
        c.bench_function("nar (mutable) 64k",
                         |b| b.iter(|| nar_mut(65536)));
    }

    pub fn aligned_read(size : usize) {

        let rows = 19;
        let cols = size;

        // let fill_vec = Vec::with_capacity(19 * size + 8)
        let fill_list = (0u8..=255).cycle().take(rows * cols + 8);
        let mut fill_vec : Vec<u8> = fill_list.collect();

        // copy elements from 0..8 to end..end + 8
        for index in 0..8 {
            fill_vec[rows * cols + index] = fill_vec[index]
        }

        // call the function
        let mut index = 0;
        let mut mod_index = 0;
        let slice = &fill_vec[0..rows * cols + 8];
        let mut ra_size = 0;
        let mut ra = NativeSimd::zero_vector();
        for _times in 0..size {
            unsafe {
                let _res = NativeSimd::read_next(
                    &mut mod_index,
                    &mut index,
                    slice,
                    size,
                    &mut ra_size,
                    &mut ra);
            }
        }
    }

    pub fn bench_aligned_read_4k(c: &mut Criterion) {
        c.bench_function("nar (aligned) 4k",
                         |b| b.iter(|| aligned_read(4096)));
    }

    pub fn bench_aligned_read_16k(c: &mut Criterion) {
        c.bench_function("nar (aligned) 16k",
                         |b| b.iter(|| aligned_read(16384)));
    }

    pub fn bench_aligned_read_64k(c: &mut Criterion) {
        c.bench_function("nar (aligned) 64k",
                         |b| b.iter(|| aligned_read(65536)));
    }


    // Now that I have new matrix code in arm_vmull, I have to bench
    // it here.

    // case where xform's k is a multiple of simd width
    // choose 8 for the sake of benchmarking
    fn arm_matrix_mul_multiple(cols : usize) {
        unsafe {
            let identity = [
                1,0,0,0, 0,0,0,0,
                0,1,0,0, 0,0,0,0,
                0,0,1,0, 0,0,0,0,
                0,0,0,1, 0,0,0,0,
                0,0,0,0, 1,0,0,0,
                0,0,0,0, 0,1,0,0,
                0,0,0,0, 0,0,1,0,
                0,0,0,0, 0,0,0,1,
            ];
            let transform = Matrix::new(8,8,true);
            transform.fill(&identity[..]);

            let mut input = Matrix::new(8,cols,false);

            // create a vector with elements 1..255 repeating
            let src : Vec<u8> = (1u8..=255).collect();
            let iter = src.into_iter().cycle().take(8*cols);
            let vec : Vec<u8> = iter.collect::<Vec<_>>();
            
            // let vec : Vec<u8> = (1u8..=9 * cols).collect();
            input.fill(&vec[..]);

            let mut output = Matrix::new(8,cols,false);

            // works if output is stored in colwise format
            new_simd_warm_multiply(&mut transform, &mut input, &mut output);
            // array has padding, so don't compare that
            assert_eq!(output.array[0..8*cols], vec);
        }
    }

    fn bench_arm_matrix_mul_16385(c: &mut Criterion) {
    c.bench_function("arm matrix mul multiple 9x16385",
                     |b| b.iter(|| arm_matrix_mul_multiple(16385)));
    }

    // Compare with reference mul of same dimensions
    fn ref_matrix_mul_multiple(cols : usize) {

        let f = new_gf8(0x11b, 0x1b);
        let identity = [
                1,0,0,0, 0,0,0,0,
                0,1,0,0, 0,0,0,0,
                0,0,1,0, 0,0,0,0,
                0,0,0,1, 0,0,0,0,
                0,0,0,0, 1,0,0,0,
                0,0,0,0, 0,1,0,0,
                0,0,0,0, 0,0,1,0,
                0,0,0,0, 0,0,0,1,
        ];
        let mut xform = Matrix::new(8,8,true);
        xform.fill(&identity[..]);

        let mut input = Matrix::new(8,cols,false);

        // create a vector with elements 1..255 repeating
        let src : Vec<u8> = (1u8..=255).collect();
        let iter = src.into_iter().cycle().take(8*cols);
        let vec : Vec<u8> = iter.collect::<Vec<_>>();
        // eprintln!("Vector length is {}", vec.len());

        //      let vec : Vec<u8> = (1u8..=9 * 17).collect();
        input.fill(&vec[..]);

        // output layout does matter for final assert!()
        let mut output = Matrix::new(8,cols, false);


        // code that was here moved into fn in main module
        
        reference_matrix_multiply(&mut xform, &mut input, &mut output, &f);
        
        // array has padding, so don't compare that
        assert_eq!(output.array[0..9*cols], vec);

    }

    fn bench_ref_gf8_matrix_mul_16385(c: &mut Criterion) {
        c.bench_function("ref matrix mul multiple 9x16385",
                         |b| b.iter(|| ref_matrix_mul_multiple(16385)));
    }

}

// bring the above into scope
#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_vmull"))]
pub use arm_ops::*;



// and then run the benchmarks
#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_vmull"))]
criterion_group!(benches,
                 // 0.1.13
                 bench_nar_tuple_4k,
//                 bench_nar_mut_4k,
                 bench_aligned_read_4k,
                 bench_nar_tuple_16k,
//                 bench_nar_mut_16k,
                 bench_aligned_read_16k,
                 bench_nar_tuple_64k,
//                 bench_nar_mut_64k,
                 bench_aligned_read_64k,
                 bench_ref_gf8_matrix_mul_16385,
                 bench_arm_gf8_matrix_mul_16385,
                 
);

#[allow(unused)]
fn dummy(_c: &mut Criterion) {
        
}


#[cfg(not(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_vmull")))]
criterion_group!(benches,dummy);


criterion_main!(benches);
