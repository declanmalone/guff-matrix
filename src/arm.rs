#![feature(asm)]


// https://pastebin.com/uRuPq2VE (assembly)
//
// C version using intrinsics:
//
// void simd_mull_reduce_poly8x8(poly8x8_t *result,
//                            poly8x8_t *a, poly8x8_t *b) {
//
//  // do non-modular poly multiply
//  poly16x8_t working = vmull_p8(*a,*b);
//
//  // copy result, and shift right
//  uint16x8_t top_nibble = vshrq_n_u16 ((uint16x8_t) working, 12);
//
//  // was uint8x16_t, but vtbl 
//  static uint8x8x2_t u4_0x11b_mod_table =  {
//    0x00, 0x1b, 0x36, 0x2d, 0x6c, 0x77, 0x5a, 0x41,
//    0xd8, 0xc3, 0xee, 0xf5, 0xb4, 0xaf, 0x82, 0x99,
//  };
//
//
//  uint8x8_t reduced = vmovn_u16(top_nibble);
//
//  // now we should have what we need to do 8x8 table lookups
//  uint8x8_t lut = vtbl2_u8(u4_0x11b_mod_table, reduced);
//
//  // Next, have to convert u8 to u16, shifting left 4 bits
//  poly16x8_t widened = (poly16x8_t) vmovl_u8(lut);
//
//  // uint16x8_t vshlq_n_u16 (uint16x8_t, const int)
//  // Form of expected instruction(s): vshl.i16 q0, q0, #0
//  widened = (poly16x8_t) vshlq_n_u16((uint16x8_t) widened, 4);
//
//  // uint16x8_t veorqq_u16 (uint16x8_t, uint16x8_t)
//  // Form of expected instruction(s): veorq q0, q0, q0
//  working = (poly16x8_t) veorq_u16((uint16x8_t) working, (uint16x8_t) widened);
//
//  // First LUT complete... repeat steps
//  
//  // extra step to clear top nibble
//  top_nibble = vshlq_n_u16 ((uint16x8_t) working, 4);
//   // to get at the one to its right
//   top_nibble = vshrq_n_u16 ((uint16x8_t) top_nibble, 12);
//   reduced = vmovn_u16(top_nibble);
//   lut = vtbl2_u8(u4_0x11b_mod_table, reduced);
//   widened = (poly16x8_t) vmovl_u8(lut);
//   // remove step, since we're applying to low byte
//   // widened = (poly16x8_t) vshlq_n_u16((uint16x8_t) widened, 4);
//   working = (poly16x8_t) veorq_u16((uint16x8_t) working, (uint16x8_t) widened);
// 
//   // apply mask (vand expects 2 registers, so use shl, shr combo)
//   //  working = (poly16x8_t) vshlq_n_u16 ((uint16x8_t) working, 8);
//   //  working = (poly16x8_t) vshrq_n_u16 ((uint16x8_t) working, 8);
// 
//   // use narrowing mov to send back result
//   *result = (poly8x8_t) vmovn_u16((uint16x8_t) working);
// }

use core::arch::arm::*;

// looking at https://doc.rust-lang.org/core/arch/arm/
//
// all sorts of intrinsics are missing... not just vmull
//
// * vmull: no
// * vmovn_u16: ok
// * vmovl_u8: ok
// * vtbl2_u8: no (no vtbl instructions at all)
// * veorq_u16: ok
// * vshrq_n_u16: no (only vshrq_n_u8)
// * vshlq_n_u16: no (only vshlq_n_u8)
//  
// Wait! https://docs.rs/core_arch/0.1.5/core_arch/aarch64/index.html
//
// These are listed under aarch64, even though I can use those
// intrinsics on a 32-bit neon machine.


fn simd_mull_reduce_poly8x8(result : &mut poly8x8_t,
			    a : &poly8x8_t, b: &poly8x8_t) {
    
    let mut working : poly16x8_t;
    let     lut     : uint8x8x2_t = {
	0x00, 0x1b, 0x36, 0x2d, 0x6c, 0x77, 0x5a, 0x41,
	0xd8, 0xc3, 0xee, 0xf5, 0xb4, 0xaf, 0x82, 0x99,
    };
    unsafe {
	asm!( "vmull.p8  {3}, {1}, {2}
	       vld1.64   {{ d16-d17 }}, [{4}]  @ load lut
              ",
	       out(reg) result,
	       in(reg) a,
	       in(reg) b,
	       in(reg) working,
	       
	);

               "vshr.u16  q9, q10, #12
	       vmovn.i16 d18, q9
	       vtbl.8    d18, {{ d16, d17 }}, d18 @ first lut
	       vmovl.u8  q9, d18
	       vshl.i16  q9, q9, #4
	       veor      q9, q10, q9
	       vshl.i16  q10, q9, #4
	       vshr.u16  q10, q10, #12
	       vmovn.i16 d20, q10
	       vtbl.8    d16, {{ d16, d17 }}, d20 @ second lut
	       vmovl.u8  q8, d16
	       veor      q8, q8, q9
	       vshl.i16  q8, q8, #8
	       vshr.u16  q8, q8, #8
	       @ store result
	       vstr      d16, [sp, #32]
	       vstr      d17, [sp, #40]
	       ";

    
}

fn random_example() {
    
    unsafe {
	asm!("vmull.p8  {}, {}, {}", inout(reg) x, number = const 5);
    }
    assert_eq!(x, 8);
}
