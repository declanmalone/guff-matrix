//! x86_64-specific SIMD

// x86 stuff is in stable
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;




// multiply
pub unsafe fn vmul_f8x16(mut a : __m128i, b : __m128i, poly : u8) -> __m128i {

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


