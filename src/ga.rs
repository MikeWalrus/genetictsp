use anyhow::Result;
use itertools::Itertools;
use rand::{
    distributions::WeightedIndex,
    prelude::*,
};
use rayon::prelude::*;
use ringbuffer::*;

pub struct Population<T: Individual> {
    individuals: Vec<T>,
    num: usize,
    num_children: usize,
    num_reserve: usize,
    mutation_probability: f64,
    history: AllocRingBuffer<(f64, f64)>,
}

pub trait Individual: Clone + Sync + Send {
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
        }
    }

    pub fn evolve(&mut self) -> Result<()> {
        self.individuals
            .sort_unstable_by(|x, y| x.fitness().partial_cmp(&y.fitness()).unwrap());
        let mut weights: Vec<usize> = Vec::with_capacity(self.num);
        weights.extend(0..self.individuals.len());
        let dist = WeightedIndex::new(weights).unwrap();
        let mut parents = dist
            .sample_iter(thread_rng())
            .map(|index| &self.individuals[index]);
        let mut children: Vec<T> = parents
            .by_ref()
            .take(self.num_children * 2)
            .tuples()
            .collect_vec()
            .par_iter()
            .map(|(p1, p2)| self.get_child(p1, p2))
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

    fn get_child(&self, p1: &T, p2: &T) -> T {
        let mut rng = thread_rng();
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
