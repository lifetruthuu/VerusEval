use vstd::prelude::*;

verus! {

fn target(x: i32) -> (result: i32)
    requires
        x >= 0,
    ensures
        result == x,
{
    x
}

}

fn main() {}
