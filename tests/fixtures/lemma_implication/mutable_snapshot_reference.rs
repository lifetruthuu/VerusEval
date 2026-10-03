use vstd::prelude::*;

verus! {

fn target(data: &mut Vec<i32>)
    requires
        old(data)@.len() > 0,
    ensures
        data@.len() == old(data)@.len(),
        forall|i: int| 0 <= i < data@.len() ==> data@[i] == old(data)@[i],
{
}

}

fn main() {}
