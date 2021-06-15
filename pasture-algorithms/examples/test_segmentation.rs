use std::{fs::File, io::BufWriter};

use pasture_algorithms::segmentation::{ransac_line, ransac_plane};
use pasture_core::{attributes_mut, containers::{PerAttributeVecPointStorage}, layout::{PointType, attributes::{INTENSITY, POSITION_3D}}, nalgebra::Vector3};
use pasture_derive::PointType;
use pasture_io::{base::PointWriter, las::LASWriter, las_rs::Builder};
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[repr(C)]
#[derive(PointType, Debug)]
pub struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn main() -> () {
    let mut buffer = PerAttributeVecPointStorage::new(SimplePoint::layout());

    //generate random points for the pointcloud
    let points: Vec<SimplePoint> = (0..20000)
        .into_par_iter()
        .map(|p| {
            let mut rng = rand::thread_rng();
            //generate plane points (along x- and y-axis)
            let mut point = SimplePoint {
                position: Vector3::new(rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0), 1.0),
                intensity: 1,
            };
            //generate z-axis points for the line
            if p % 4 == 0 {
                point.position = Vector3::new(0.0, 0.0, rng.gen_range(0.0..200.0));
            }
            //generate outliers
            if p % 50 == 0 {
                point.position.z = rng.gen_range(-50.0..50.2);
            }
            point
        })
        .collect();

    buffer.push_points(&points);
    println!("done generating pointcloud");
    let plane_and_points = ransac_plane::<PerAttributeVecPointStorage>(&buffer, 0.01, 50, true);
    println!("done ransac_plane");
    let line_and_points = ransac_line::<PerAttributeVecPointStorage>(&buffer, 0.01, 50, true);
    println!("done ransac_line");
    println!("{:?}", plane_and_points.0);
    println!("{:?}", line_and_points.0);

    
    // change intensity for the line and plane points
    for (index, p) in attributes_mut![&POSITION_3D => Vector3<f64>, &INTENSITY => u16, &mut buffer].enumerate() {
        if line_and_points.1.contains(&index) {
            *p.1 = 500;
        } else if plane_and_points.1.contains(&index) {
            *p.1 = 700;
        } else {
            *p.1 = 300;
        }
    }
    println!("changed intensity");

    // write into file
    let writer = BufWriter::new(File::create("testCloud.las").unwrap());
    let header = Builder::from((1, 4)).into_header().unwrap();
    let mut writer = LASWriter::from_writer_and_header(writer, header, false).unwrap();
    writer.write(&buffer).unwrap();
}
