use vstd::prelude::*;

verus! {

spec fn pred(x: int) -> bool {
    x >= 0
}

fn target(x: int)
    requires
        pred(x),
{
}

}

fn main() {}
