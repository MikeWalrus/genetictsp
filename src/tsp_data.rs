use std::collections::HashMap;

use tspf::metric::euc_2d;

pub trait Tsp: Sync {
    fn dim(&self) -> usize;

    fn weight(&self, a: usize, b: usize) -> f64;
}
pub struct Matrix {
    dim: usize,
    matrix: Vec<f64>,
}

impl Tsp for Matrix {
    fn dim(&self) -> usize {
        self.dim
    }

    fn weight(&self, a: usize, b: usize) -> f64 {
        self.matrix[a * self.dim + b]
    }
}

impl Matrix {
    pub fn new_lower_diag_row(matrix: &[Vec<f64>], dim: usize) -> Self {
        let mut v: Vec<f64> = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for j in 0..dim {
                if i < j {
                    v.push(matrix[j][i])
                } else {
                    v.push(matrix[i][j])
                }
            }
        }
        Self { dim, matrix: v }
    }
}

pub struct Euc2d {
    dim: usize,
    coords: Vec<[f64; 2]>,
}

impl Tsp for Euc2d {
    fn dim(&self) -> usize {
        self.dim
    }

    fn weight(&self, a: usize, b: usize) -> f64 {
        euc_2d(&self.coords[a], &self.coords[b])
    }
}

impl Euc2d {
    pub fn new(coords: &HashMap<usize, tspf::Point>, dim: usize) -> Self {
        let mut v: Vec<[f64; 2]> = vec![[0f64, 0f64]; dim];
        for (&index, coord) in coords {
            let coord = coord.pos();
            v[index - 1][0] = coord[0];
            v[index - 1][1] = coord[1];
        }
        Self { dim, coords: v }
    }
}
