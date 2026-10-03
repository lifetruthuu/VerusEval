use vstd::prelude::*;
verus! {
fn identity(x: u64) -> (r: u64)
    requires x < 100,
    ensures r == x,
{
    x
}
}
fn main() {}
