use pasture_algorithms::{cluster_extraction::extract_clusters_euclidean, data_structures::kdtree};
use pasture_core::{containers::PerAttributeVecPointStorage, layout::PointType, nalgebra::Vector3};
use pasture_derive::PointType;

use pasture_io::{base::PointReader, las::LASReader};

#[repr(C)]
#[derive(PointType, Debug)]
pub struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}
fn main() -> () {
    //is this how to read in LAS-files, can i not get a PerAttributeVecPointStorage?
    // let mut reader = Reader::from_path("/var/home/raban/Nextcloud/Uni/vc-praktikum/Rust/second/pasture/pasture-io/resources/test/10_points_format_0.las").unwrap();
    let mut reader = LASReader::from_path("/var/home/raban/Nextcloud/Uni/vc-praktikum/Rust/second/pasture/pasture-io/resources/test/10_points_format_0.las").unwrap();
    let mut buffer = PerAttributeVecPointStorage::new(SimplePoint::layout());
    reader
        .read_into(
            &mut buffer,
            reader.get_metadata().number_of_points().unwrap(),
        )
        .unwrap();
    // for wrapped_point in reader.points() {
    //     let point = wrapped_point.unwrap();
    //     buffer.push_point(SimplePoint {
    //         position: Vector3::new(point.x, point.y, point.z),
    //         intensity: point.intensity,
    //     });
    // }
    // let mut points = vec![
    //     SimplePoint {
    //         position: Vector3::new(1.0, 2.0, 3.0),
    //         intensity: 42,
    //     },
    //     SimplePoint {
    //         position: Vector3::new(-1.0, -2.0, -3.0),
    //         intensity: 84,
    //     },
    // ];
    // let mut rng = rand::thread_rng();

    // let i: i32 = 0;
    // for i in i..2000 {
    //     points.push(SimplePoint {
    //         position: Vector3::new(
    //             rng.gen_range(0.0..100.0),
    //             rng.gen_range(0.0..100.0),
    //             rng.gen_range(0.0..100.0),
    //         ),
    //         intensity: i as u16,
    //     })
    // }
    // buffer.push_points(&points);

    let tree = kdtree::kdtree_from_buffer(&mut buffer);
    let found = tree.nearest(&[23.0, 122.0, 1.0]).unwrap();
    println!("found closest point: {:?}", found);
    extract_clusters_euclidean(&tree);
}
