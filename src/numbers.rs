
/// Greatest Common Divisor for two integers
pub fn gcd(mut a : usize, mut b : usize) -> usize {
  let mut t;
    loop {
	if b == 0 { return a }
	t = b;
	b = a % b;
	a = t;
    }
}

/// Greatest Common Divisor for three integers
pub fn gcd3(a : usize, b : usize, c: usize) -> usize {
    gcd(a, gcd(b,c))
}

/// Greatest Common Divisor for four integers
pub fn gcd4(a : usize, b : usize, c: usize, d : usize) -> usize {
    gcd(gcd(a,b), gcd(c,d))
}

/// Least Common Multiple for two integers
pub fn lcm(a : usize, b : usize) -> usize {
    (a / gcd(a,b)) * b
}

/// Least Common Multiple for three integers
pub fn lcm3(a : usize, b : usize, c: usize) -> usize {
    lcm( lcm(a,b), c)
}

/// Least Common Multiple for four integers
pub fn lcm4(a : usize, b : usize, c: usize, d : usize) -> usize {
    lcm( lcm(a,b), lcm(c,d) )
}

