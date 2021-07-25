/* C code...



void simd_long_mul_reduce_buffer(uint8_t poly,
                                 uint8_t *d, uint8_t *aa, uint8_t *bb,
                                 long bytes) {

  if (bytes & 15) {
    fprintf(stderr, "bytes must be a multiple of 16\n");
    exit(1);
  }
  int times = bytes >> 4;

#ifdef DEBUG
  static int pp = 3;
#endif

  poly8x16_t zero = vmovq_n_p8(0); // there's also a hardware zero vec
  poly8x16_t high = vmovq_n_p8(128);
  poly8x16_t vpol = vmovq_n_p8(poly);
  poly8x16_t a,b;               // don't clobber *aa, keep b in reg
  uint8x16_t mask;
  uint8x16_t res, bool, temp;

  while (times--) {
    mask = vmovq_n_u8(1);
    a = *(poly8x16_t*) aa; aa += 16;
    b = *(poly8x16_t*) bb; bb += 16;

#ifdef DEBUG
    if (pp) {
      pp_vector("A   ", &a);
      pp_vector("B   ", &b);
      // pp_vector("Mask", &mask);
    }
#endif
    
    
    bool = vtstq_u8 ((uint8x16_t) b, mask);
    res  = (uint8x16_t) vbslq_p8 (bool, a, zero);

    signed count = 7;
    do {
      bool = vtstq_u8 ((uint8x16_t) a, (uint8x16_t) high);
      a    = (poly8x16_t) vshlq_n_u8((uint8x16_t)a, 1);
      temp = (uint8x16_t) veorq_u8((uint8x16_t) a, (uint8x16_t) vpol);
      a    =              vbslq_p8 (bool, (poly8x16_t) temp, a);

      mask = (uint8x16_t) vshlq_n_u8((uint8x16_t) mask, 1);
      bool = (uint8x16_t) vtstq_u8 ((uint8x16_t) b, mask);
      temp = (uint8x16_t) veorq_u8((uint8x16_t) res, (uint8x16_t) a);
      res  = (uint8x16_t) vbslq_p8 (bool, (poly8x16_t)temp, (poly8x16_t)res);
    } while (--count);

#ifdef DEBUG
    if (pp) {
      pp_vector("RES ", &res);
      pp--;
    }
#endif
    *(poly8x16_t*) d  = (poly8x16_t)res;
    d += 16;
  }
}

 */

