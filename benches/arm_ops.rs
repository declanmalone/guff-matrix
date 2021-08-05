
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use criterion::BenchmarkId;



#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_vmull"))]
pub mod arm_ops {

    use guff::{GaloisField, new_gf8, F8 };
    use guff_matrix::*;
    use guff_matrix::arm_vmull::*;

    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    use criterion::BenchmarkId;


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

    pub fn bench_ref_gf8_vec(c: &mut Criterion) {
        c.bench_function("ref gf8 vec tmp", |b| b.iter(|| ref_gf8_vec(32768)));
    }

    pub fn bench_simd_gf8_vec(c: &mut Criterion) {
        c.bench_function("simd gf8 vec tmp", |b| b.iter(|| simd_gf8_vec(32768)));
    }


}

// bring the above into scope
#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_vmull"))]
pub use arm_ops::*;

// and then run the benchmarks
#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_vmull"))]
criterion_group!(benches,
                 // 0.1.13
                 bench_ref_gf8_vec,
                 bench_simd_gf8_vec,
);

fn dummy(c: &mut Criterion) {
        
    }


#[cfg(not(all(any(target_arch = "aarch64", target_arch = "arm"), feature = "arm_vmull")))]
criterion_group!(benches,dummy);


criterion_main!(benches);
