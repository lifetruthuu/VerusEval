use vstd::prelude::*;

verus! {

fn target(values: &mut Vec<i32>)
    requires
        old(values)@.len() > 0,
    ensures
        values@.len() == old(values)@.len(),
        forall|index: int| 0 <= index < values@.len() ==> values@[index] == old(values)@[index],
{
}

}

fn main() {}
