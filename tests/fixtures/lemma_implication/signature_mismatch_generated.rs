use vstd::prelude::*;

verus! {

fn target(x: int, y: int)
    requires
        x >= y,
{
}

}

fn main() {}
