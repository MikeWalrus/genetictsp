use std::{iter::repeat, os::unix::process::parent_id};

use anyhow::Result;
use itertools::Itertools;
use rand::{
    distributions::WeightedIndex,
    prelude::{Distribution, SliceRandom},
    thread_rng, Rng,
};
use rayon::prelude::*;
use ringbuffer::*;

pub struct Population<T: Individual> {
    individuals: Vec<T>,
    num: usize,
    num_children: usize,
    num_reserve: usize,
    rand_amount: usize,
    mutation_probability: f64,
    history: AllocRingBuffer<(f64, f64)>,
}

pub trait Individual: Clone {
    fn fitness(&self) -> f64;
    fn update_fitness(&mut self);
    fn mutate<R: Rng>(&mut self, rng: &mut R);
    fn crossover<R: Rng>(p1: &Self, p2: &Self, rng: &mut R) -> Self;
}

impl<T: Individual> Population<T> {
    pub fn new(
        individuals: Vec<T>,
        num: usize,
        crossover_probability: f64,
        mutation_probability: f64,
    ) -> Self {
        let num_children = (num as f64 * crossover_probability) as usize;
        Population {
            individuals,
            num,
            num_children,
            num_reserve: num - num_children,
            mutation_probability,
            history: AllocRingBuffer::with_capacity(8),
            rand_amount: num_children + num,
        }
    }

    pub fn evolve(&mut self) -> Result<()> {
        let mut rng = thread_rng();
        let mut weights: Vec<f64> = Vec::with_capacity(self.num);
        let min = self.history.back().map(|(_, min)| *min).unwrap_or(0f64) * 0.9;
        weights.extend(self.individuals.iter().map(|i| i.fitness() - min));
        let dist = WeightedIndex::new(weights).unwrap();
        let mut parents = dist
            .sample_iter(thread_rng())
            .map(|index| &self.individuals[index]);
        let mut children: Vec<T> = parents
            .by_ref()
            .take(self.num_children * 2)
            .tuples()
            .map(|(p1, p2)| self.get_child(p1, p2, &mut rng))
            .collect();
        children.extend(parents.take(self.num_reserve).map(T::clone));
        let mut fitness_max = 0f64;
        let mut fitness_min = f64::MAX;
        for i in &mut children {
            i.update_fitness();
            if fitness_max < i.fitness() {
                fitness_max = i.fitness()
            }
            if fitness_min > i.fitness() {
                fitness_min = i.fitness()
            }
        }
        self.individuals = children;
        self.history.push((fitness_max, fitness_min));
        Ok(())
    }

    fn get_child<R: Rng>(&self, p1: &T, p2: &T, mut rng: R) -> T {
        let mut i = Individual::crossover(p1, p2, &mut rng);
        if rng.gen_bool(self.mutation_probability) {
            i.mutate(&mut rng)
        }
        i
    }

    /// Get a reference to the population's history.
    pub fn history(&self) -> &AllocRingBuffer<(f64, f64)> {
        &self.history
    }
}
