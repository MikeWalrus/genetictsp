use std::{collections::HashSet, iter::repeat, path::Path};

use super::ga::*;
use rand::{
    distributions::Uniform,
    prelude::{Distribution, SliceRandom},
    thread_rng, Rng,
};
use ringbuffer::*;
use tspf::*;

#[derive(Clone)]
struct Spec<'a> {
    tsp: &'a Tsp,
    num_crossover_points: usize,
    num_mutation_points: usize,
}

struct Edge(usize, usize);
impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1 || self.0 == other.1 && self.1 == other.0
    }
}

impl Eq for Edge {}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == other {
            Some(std::cmp::Ordering::Equal)
        } else {
            (self.0, self.1).partial_cmp(&(other.0, other.1))
        }
    }
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self == other {
            std::cmp::Ordering::Equal
        } else {
            (self.0, self.1).cmp(&(other.0, other.1))
        }
    }
}

#[derive(Clone)]
struct Route<'a> {
    route: Vec<usize>,
    fitness: f64,
    spec: &'a Spec<'a>,
}

impl<'a> Individual for Route<'a> {
    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn mutate<R: Rng>(&mut self, mut rng: &mut R) {
        let range: Uniform<usize> = Uniform::from(1..self.spec.tsp.dim() - 1);
        let mut mutation_indexes = Vec::with_capacity(self.spec.num_mutation_points);
        mutation_indexes.extend(
            range
                .sample_iter(&mut rng)
                .take(self.spec.num_mutation_points),
        );
        let mut mutation_points: Vec<usize> = mutation_indexes
            .iter()
            .copied()
            .map(|i| self.route[i])
            .collect();
        mutation_points.shuffle(rng);
        mutation_indexes
            .into_iter()
            .zip(mutation_points.iter())
            .for_each(|(index, &new_value)| self.route[index] = new_value);
    }

    fn crossover<R: Rng>(p1: &Self, p2: &Self, rng: &mut R) -> Self {
        let range: Uniform<usize> = Uniform::from(1..p1.spec.tsp.dim());
        let mut crossover_points = Vec::with_capacity(p1.spec.num_crossover_points + 2);
        crossover_points.extend(range.sample_iter(rng).take(p1.spec.num_crossover_points));
        crossover_points.push(1);
        crossover_points.push(p1.spec.tsp.dim() - 1);
        crossover_points.sort_unstable();
        let segments = crossover_points.windows(2);
        let gene_from_p1: HashSet<usize> = segments
            .clone()
            .step_by(2)
            .flat_map(|window| &p1.route[window[0]..window[1]])
            .copied()
            .collect();
        let mut child = p1.clone();
        let mut gene_from_p2 = p2.route[1..p2.route.len() - 1]
            .iter()
            .filter(|&i| !gene_from_p1.contains(i))
            .copied();
        segments.skip(1).step_by(2).for_each(|window| {
            for i in &mut child.route[window[0]..window[1]] {
                *i = gene_from_p2.next().unwrap()
            }
        });
        //println!("{:?}", crossover_points);
        //println!("{:?}\n{:?}\n{:?}\n", p1.route, p2.route, child.route);
        child
    }

    fn update_fitness(&mut self) {
        self.fitness = 1f64 / total_weight(&self.route, self.spec.tsp)
    }
}

fn total_weight(route: &[usize], tsp: &Tsp) -> f64 {
    route
        .windows(2)
        .map(|edge| tsp.weight(edge[0], edge[1]))
        .sum()
}

pub fn solve_tsp(path: &Path) {
    let tsp = TspBuilder::parse_path(path).unwrap();
    let spec = Spec {
        tsp: &tsp,
        num_crossover_points: 4,
        num_mutation_points: 5,
    };
    let mut init_route: Vec<usize> = Vec::with_capacity(tsp.dim() + 2);
    init_route.extend(0..tsp.dim());
    init_route.push(0);

    let num = 1000;
    let mut rng = thread_rng();
    let individuals = repeat(init_route.clone())
        .take(num)
        .map(|mut i| {
            i[1..tsp.dim() - 1].shuffle(&mut rng);
            Route {
                fitness: 1f64 / total_weight(&i, &tsp),
                spec: &spec,
                route: i,
            }
        })
        .collect();

    let mut population = Population::new(individuals, num, 0.5, 0.1);
    for _ in 0..1000000 {
        population.evolve().unwrap();
        println!("{:?}", population.history().peek().unwrap())
    }
}
