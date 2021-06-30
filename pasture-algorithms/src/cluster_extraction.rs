use kd_tree::KdTree;
use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;

use crate::data_structures::kdtree::Item;
#[repr(C)]
#[derive(PointType, Debug)]
pub struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

pub fn extract_clusters_euclidean(tree: &KdTree<Item>) -> Vec<Vec<&Item>> {
    let mut c: Vec<Vec<&Item>> = vec![];
    let mut q: Vec<&Item> = vec![];
    let mut processed: Vec<&Item> = vec![];
    let mut counter = 0;
    for p in tree.iter() {
        q.push(p);
        while processed.len() < tree.len() && counter < q.len() {
            let set = tree.within_radius(q[counter], 15.0);
            for i in set {
                if !q.contains(&i) {
                    q.push(&i);
                    // println!("pushed stuff");
                }
                processed.push(i);
            }
            counter += 1;
        }
        c.push(q.clone());
        q = vec![];
        processed = vec![];
        counter = 0;
    }
    println!("got clusters: {:?}", c.len());
    c
}
