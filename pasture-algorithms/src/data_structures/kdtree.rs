use kd_tree::{KdPoint, KdTree};
use pasture_core::{
    containers::{PerAttributeVecPointStorage, PointBufferExt},
    layout::attributes::POSITION_3D,
    nalgebra::Vector3,
};

#[derive(Debug)]
pub struct Item {
    x: f64,
    y: f64,
    z: f64,
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl KdPoint for Item {
    type Scalar = f64;
    type Dim = typenum::U3;
    fn at(&self, k: usize) -> f64 {
        if k == 0 {
            self.x
        } else if k == 1 {
            self.y
        } else {
            self.z
        }
    }
}

pub fn kdtree_from_buffer(buffer: &mut PerAttributeVecPointStorage) -> KdTree<Item> {
    let vecbuf: Vec<Item> = buffer
        .iter_attribute::<Vector3<f64>>(&POSITION_3D)
        .map(|pos| Item {
            x: pos.x,
            y: pos.y,
            z: pos.z,
        })
        .collect();

    let kdtree = KdTree::build_by_ordered_float(vecbuf);
    return kdtree;
}
