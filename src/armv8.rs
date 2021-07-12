
// This file name will change.
//
// There's something strange about how Rust breaks up arches into arm
// and aarch64. All of the NEON instructions are declared in the
// latter module, even though many/most of them are available on armv7
// boards.
//
// I'll stick with calling this file armv8 for now, since I want to
// use the intrinsics provided in arch::aarch64, and I have an Aarch64
// board that I can test it on.
//
// Later, I'll write a pure assembly version of this and inline with
// the asm! macro. Then I won't be dependent on using the intrinsics
// in arch::aarch64.

