// Things that make armv8 different from armv7
//
// * tbl instructions that operate on 128-bit vectors (eg, vqtbl4q_u8)
// * increased latency on tbl2, tbl3, tbl4 instructions!
// * more NEON registers (32 vs 16)
// * polynomial multiply on 128-bit vectors (vmull_high_p8)
//
// See:
//
// https://www.cnx-software.com/2017/08/07/how-arm-nerfed-neon-permute-instructions-in-armv8/
//
// The 

