use std::vec;

use pasture_core::{
    containers::{PointBuffer, PointBufferExt},
    layout::attributes::POSITION_3D,
    nalgebra::Vector3,
};
use rand::Rng;
use rayon::prelude::*;

/// Represents a line between two points
/// the ranking shows how many points of the pointcloud are inliers for this specific line
#[derive(Debug)]
pub struct Line {
    first: Vector3<f64>,
    second: Vector3<f64>,
    ranking: usize,
}

/// Represents a plane in coordinate-form: ax + by + cz + d = 0
/// the ranking shows how many points of the pointcloud are inliers for this specific plane
#[derive(Debug)]
pub struct Plane {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    ranking: usize,
}

/// Ransac Plane Segmentation.
///
/// Returns the plane with the highest rating/most inliers and the indices of the inliers.
///
/// * `buffer` - The pointcloud-buffer.
/// * `distance_threshold` - The maximum distance that a point is counted as an inlier.
/// * `num_of_iterations` - The number of iterations the algorithm performs.
/// * `parallel` - if true: runs in parallel (using rayon)
pub fn ransac_plane<T: PointBuffer + Sync>(
    buffer: &T,
    distance_threshold: f64,
    num_of_iterations: usize,
    parallel: bool,
) -> (Plane, Vec<usize>) {
    if parallel {
        return ransac_plane_par(buffer, distance_threshold, num_of_iterations);
    } else {
        return ransac_plane_serial(buffer, distance_threshold, num_of_iterations);
    }
}

/// Ransac Line Segmentation.
///
/// Returns the line with the highest rating/most inliers and the indices of the inliers.
///
/// * `buffer` - The pointcloud-buffer.
/// * `distance_threshold` - The maximum distance that a point is counted as an inlier.
/// * `num_of_iterations` - The number of iterations the algorithm performs.
/// * `parallel` - if true: runs in parallel (using rayon)
pub fn ransac_line<T: PointBuffer + Sync>(
    buffer: &T,
    distance_threshold: f64,
    num_of_iterations: usize,
    parallel: bool,
) -> (Line, Vec<usize>) {
    if parallel {
        return ransac_line_par(buffer, distance_threshold, num_of_iterations);
    } else {
        return ransac_line_serial(buffer, distance_threshold, num_of_iterations);
    }
}

/// calculates the distance between a point and a plane
fn distance_point_plane(point: &Vector3<f64>, plane: &Plane) -> f64 {
    let d = (plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d).abs();
    let e = (plane.a * plane.a + plane.b * plane.b + plane.c * plane.c).sqrt();
    return d / e;
}

/// calculates the distance between a point and a line
/// careful: seems to be slow in debug, but is really fast in release
fn distance_point_line(point: &Vector3<f64>, line: &Line) -> f64 {
    (line.second - line.first)
        .cross(&(line.first - point))
        .norm()
        / (line.second - line.first).norm()
}

/// ransac plane algorithm in parallel
fn ransac_plane_par<T: PointBuffer + Sync>(
    buffer: &T,
    distance_threshold: f64,
    num_of_iterations: usize,
) -> (Plane, Vec<usize>) {
    // iterate in parallel over num_of_iterations
    (0..num_of_iterations)
        .into_par_iter()
        .map(|_x| {
            let mut rng = rand::thread_rng();
            let rand1 = rng.gen_range(0..buffer.len());
            let mut rand2 = rng.gen_range(0..buffer.len());
            while rand1 == rand2 {
                rand2 = rng.gen_range(0..buffer.len());
            }
            let mut rand3 = rng.gen_range(0..buffer.len());
            // make sure we have 3 unique random numbers to generate plane model
            while rand2 == rand3 || rand1 == rand3 {
                rand3 = rng.gen_range(0..buffer.len());
            }
            let p_a: Vector3<f64> = buffer.get_attribute(&POSITION_3D, rand1);
            let p_b: Vector3<f64> = buffer.get_attribute(&POSITION_3D, rand2);
            let p_c: Vector3<f64> = buffer.get_attribute(&POSITION_3D, rand3);

            // compute plane from the three positions
            let vec1 = p_b - p_a;
            let vec2 = p_c - p_a;
            let normal = vec1.cross(&vec2);
            let d = -normal.dot(&p_a);
            let mut curr_hypo = Plane {
                a: normal.x,
                b: normal.y,
                c: normal.z,
                d,
                ranking: 0,
            };

            // find all points that belong to the plane
            let mut current_positions = vec![];

            for (index, p) in buffer
                .iter_attribute::<Vector3<f64>>(&POSITION_3D)
                .enumerate()
            {
                let distance = distance_point_plane(&p, &curr_hypo);
                if distance < distance_threshold {
                    //we found a point that belongs to the plane
                    curr_hypo.ranking += 1;
                    current_positions.push(index);
                }
            }
            // return the current hypothesis and the corresponding positions
            (curr_hypo, current_positions)
        })
        // get the beste hypothesis from all iterations
        .max_by(|(x, _y), (a, _b)| x.ranking.cmp(&a.ranking))
        .unwrap()
}

/// ransac plane algorithm in serial
fn ransac_plane_serial<T: PointBuffer + Sync>(
    buffer: &T,
    distance_threshold: f64,
    num_of_iterations: usize,
) -> (Plane, Vec<usize>) {
    let mut best_fit = Plane {
        a: 0.0,
        b: 0.0,
        c: 0.0,
        d: 0.0,
        ranking: 0,
    };
    let mut best_positions: Vec<usize> = vec![];

    //iterate num_of_iterations times
    for _i in 0..num_of_iterations {
        let mut rng = rand::thread_rng();
        let rand1 = rng.gen_range(0..buffer.len());
        let mut rand2 = rng.gen_range(0..buffer.len());
        while rand1 == rand2 {
            rand2 = rng.gen_range(0..buffer.len());
        }
        let mut rand3 = rng.gen_range(0..buffer.len());
        // make sure we have 3 unique random numbers to generate the plane model
        while rand2 == rand3 || rand1 == rand3 {
            rand3 = rng.gen_range(0..buffer.len());
        }
        let p_a: Vector3<f64> = buffer.get_attribute(&POSITION_3D, rand1);
        let p_b: Vector3<f64> = buffer.get_attribute(&POSITION_3D, rand2);
        let p_c: Vector3<f64> = buffer.get_attribute(&POSITION_3D, rand3);

        // compute plane from the three positions
        let vec1 = p_b - p_a;
        let vec2 = p_c - p_a;
        let normal = vec1.cross(&vec2);
        let d = -normal.dot(&p_a);
        let mut curr_hypo = Plane {
            a: normal.x,
            b: normal.y,
            c: normal.z,
            d,
            ranking: 0,
        };

        // find all points that belong to the plane
        let mut current_positions = vec![];
        for (index, p) in buffer
            .iter_attribute::<Vector3<f64>>(&POSITION_3D)
            .enumerate()
        {
            let distance = distance_point_plane(&p, &curr_hypo);
            if distance < distance_threshold {
                // we found an inlier
                curr_hypo.ranking += 1;
                current_positions.push(index);
            }
        }
        // keep only the best model
        if curr_hypo.ranking > best_fit.ranking {
            best_fit = curr_hypo;
            best_positions = current_positions;
        }
    }
    // return the best model and the inliers
    (best_fit, best_positions)
}

/// ransac line algorithm in parallel
pub fn ransac_line_par<T: PointBuffer + Sync>(
    buffer: &T,
    distance_threshold: f64,
    num_of_iterations: usize,
) -> (Line, Vec<usize>) {
    // iterate num_of_iterations in parallel
    (0..num_of_iterations)
        .into_par_iter()
        .map(|_x| {
            //we need to choose two random points from the pointcloud here
            let mut rng = rand::thread_rng();
            let rand1 = rng.gen_range(0..buffer.len());
            let mut rand2 = rng.gen_range(0..buffer.len());
            //make sure we have two different points
            while rand1 == rand2 {
                rand2 = rng.gen_range(0..buffer.len());
            }
            //generate line from the two points
            let mut curr_hypo = Line {
                first: buffer.get_attribute(&POSITION_3D, rand1),
                second: buffer.get_attribute(&POSITION_3D, rand2),
                ranking: 0,
            };

            let mut curr_positions = vec![];
            // find all points that belong to the line
            for (index, p) in buffer
                .iter_attribute::<Vector3<f64>>(&POSITION_3D)
                .enumerate()
            {
                let distance = distance_point_line(&p, &curr_hypo);
                if distance < distance_threshold {
                    // we found a point of the line
                    curr_positions.push(index);
                    curr_hypo.ranking += 1;
                }
            }
            // return current line and positions
            (curr_hypo, curr_positions)
        })
        // use only the best line (highest ranking)
        .max_by(|(x, _y), (a, _b)| x.ranking.cmp(&a.ranking))
        .unwrap()
}

/// ransac line algorithm in serial
pub fn ransac_line_serial<T: PointBuffer + Sync>(
    buffer: &T,
    distance_threshold: f64,
    num_of_iterations: usize,
) -> (Line, Vec<usize>) {
    let mut best_fit = Line {
        first: Vector3::new(0.0, 0.0, 0.0),
        second: Vector3::new(0.0, 0.0, 0.0),
        ranking: 0,
    };
    let mut best_positions = vec![];
    // iterate num_of_iterations times
    for _i in 0..num_of_iterations {
        // we need to choose two random points from the pointcloud here
        let mut rng = rand::thread_rng();
        let rand1 = rng.gen_range(0..buffer.len());
        let mut rand2 = rng.gen_range(0..buffer.len());
        // make sure we have two different points
        while rand1 == rand2 {
            rand2 = rng.gen_range(0..buffer.len());
        }
        // generate line from the two points
        let mut curr_hypo = Line {
            first: buffer.get_attribute(&POSITION_3D, rand1),
            second: buffer.get_attribute(&POSITION_3D, rand2),
            ranking: 0,
        };
        let mut curr_positions = vec![];

        // find all points in the pointbuffer that belong to the line
        for (index, p) in buffer
            .iter_attribute::<Vector3<f64>>(&POSITION_3D)
            .enumerate()
        {
            let distance = distance_point_line(&p, &curr_hypo);
            if distance < distance_threshold {
                // we found a point of the line
                curr_positions.push(index);
                curr_hypo.ranking += 1;
            }
        }
        // only keep the best line-model
        if curr_hypo.ranking > best_fit.ranking {
            best_fit = curr_hypo;
            best_positions = curr_positions;
        }
    }
    // return the best line-model and corresponding point-indices
    (best_fit, best_positions)
}
