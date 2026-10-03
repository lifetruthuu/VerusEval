use vstd::prelude::*;

verus! {

fn target(n: i32) -> (res: i32)
    requires
        n >= 0,
    ensures
        res == n,
{
    n
}

}

fn main() {}
