

#[cfg(target_arch = "arm")]
use core::arch::arm::*;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

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

// OK. Checking *nightly* docs shows that all the intrinsics I need
// are there, including vmull!
//
// With that discovery, I can include assembly the easy way: using
// intrinsics throughout. No need for asm! or external libraries.





// Interleaving C version in comments

// void simd_mull_reduce_poly8x8(poly8x8_t *result,
//			      poly8x8_t *a, poly8x8_t *b) {

pub fn simd_mull_reduce_poly8x8(result : &mut poly8x8_t,
			 a : &poly8x8_t, b: &poly8x8_t) {

    unsafe {
	// // do non-modular poly multiply
	// poly16x8_t working = vmull_p8(*a,*b);
	let mut working : poly16x8_t = vmull_p8(*a, *b);

	// // copy result, and shift right
	// uint16x8_t top_nibble = vshrq_n_u16 ((uint16x8_t) working, 12);
	let top_nibble : uint16x8_t = vshrq_n_u16 (vreinterpretq_u16_p16(working), 12);

//  // was uint8x16_t, but vtbl 
//  static uint8x8x2_t u4_0x11b_mod_table =  {
//    0x00, 0x1b, 0x36, 0x2d, 0x6c, 0x77, 0x5a, 0x41,
//    0xd8, 0xc3, 0xee, 0xf5, 0xb4, 0xaf, 0x82, 0x99,
//  };

	let u4_0x11b_mod_table = uint8x8x2_t (
	    uint8x8_t (0x00, 0x1b, 0x36, 0x2d, 0x6c, 0x77, 0x5a, 0x41, ),
	    uint8x8_t (0xd8, 0xc3, 0xee, 0xf5, 0xb4, 0xaf, 0x82, 0x99, ), 
	);

	// looks like we can't get a uint16x8_t output, so have to break up
	// into two 8x8 lookups. Can we cast to access the halves?

	// Looks like we want vget high/low..  Actually, we've got 16-bit
	// values, so the correct thing to do is a mov?

	// vmovn vector move narrow.u16 should do what I want:
	//  uint8x8_t vmovn_u16 (uint16x8_t)
	// Form of expected instruction(s): vmovn.i16 d0, q0

	// These should cast 
	// uint16x8_t reduced_low  = vget_low_u16(top_nibble);
	// uint16x8_t reduced_high = vget_high_u16(top_nibble);

	//   uint8x8_t reduced = vmovn_u16(top_nibble);

	let mut reduced : uint8x8_t = vmovn_u16(top_nibble);

	// now we should have what we need to do 8x8 table lookups
	//  uint8x8_t lut = vtbl2_u8(u4_0x11b_mod_table, reduced);
	let lut : uint8x8_t = vtbl2_u8(u4_0x11b_mod_table, reduced);

  // Next, have to convert u8 to u16, shifting left 4 bits
//  poly16x8_t widened = (poly16x8_t) vmovl_u8(lut);

  // uint16x8_t vshlq_n_u16 (uint16x8_t, const int)
  // Form of expected instruction(s): vshl.i16 q0, q0, #0
//  widened = (poly16x8_t) vshlq_n_u16((uint16x8_t) widened, 4);

  // uint16x8_t veorqq_u16 (uint16x8_t, uint16x8_t)
  // Form of expected instruction(s): veorq q0, q0, q0
//  working = (poly16x8_t) veorq_u16((uint16x8_t) working, (uint16x8_t) widened);

  // First LUT complete... repeat steps
  
  // extra step to clear top nibble
//  top_nibble = vshlq_n_u16 ((uint16x8_t) working, 4);
  // to get at the one to its right
//  top_nibble = vshrq_n_u16 ((uint16x8_t) top_nibble, 12);
//  reduced = vmovn_u16(top_nibble);
//  lut = vtbl2_u8(u4_0x11b_mod_table, reduced);
//  widened = (poly16x8_t) vmovl_u8(lut);
  // remove step, since we're applying to low byte
  // widened = (poly16x8_t) vshlq_n_u16((uint16x8_t) widened, 4);
//  working = (poly16x8_t) veorq_u16((uint16x8_t) working, (uint16x8_t) widened);

  // apply mask (vand expects 2 registers, so use shl, shr combo)
  //  working = (poly16x8_t) vshlq_n_u16 ((uint16x8_t) working, 8);
  //  working = (poly16x8_t) vshrq_n_u16 ((uint16x8_t) working, 8);

  // use narrowing mov to send back result
//  *result = (poly8x8_t) vmovn_u16((uint16x8_t) working);

    }
}
