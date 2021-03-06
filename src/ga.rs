use anyhow::Result;
use itertools::Itertools;
use rand::{distributions::WeightedIndex, prelude::*};

pub struct Population<T: Individual> {
    individuals: Vec<T>,
    num: usize,
    num_crossover: usize,
    num_reserve: usize,
    mutation_probability: f64,
}

pub trait Individual: Clone + Sync + Send {
    fn fitness(&self) -> f64;
    fn update_fitness(&mut self);
    fn mutate<R: Rng>(&mut self, rng: &mut R);
    fn crossover<R: Rng>(p1: &Self, p2: &Self, rng: &mut R) -> [Self; 2];
}

impl<T: Individual> Population<T> {
    pub fn new(
        mut individuals: Vec<T>,
        num: usize,
        crossover_probability: f64,
        mutation_probability: f64,
    ) -> Self {
        let num_crossover = (num as f64 * crossover_probability) as usize / 2;
        sort_individuals(&mut individuals);
        Population {
            individuals,
            num,
            num_crossover,
            num_reserve: num - 2 * num_crossover,
            mutation_probability,
        }
    }

    pub fn evolve(&mut self) -> Result<()> {
        let mut rng = thread_rng();
        let mut weights: Vec<usize> = Vec::with_capacity(self.num);
        weights.extend(0..self.individuals.len());
        let dist = WeightedIndex::new(weights).unwrap();
        let mut parents = dist
            .sample_iter(thread_rng())
            .map(|index| &self.individuals[index]);
        let mut children: Vec<T> = Vec::with_capacity(self.num);
        children.extend(
            parents
                .by_ref()
                .take(self.num_crossover)
                .tuples()
                .collect_vec()
                .iter()
                .flat_map(|(p1, p2)| {
                    Individual::crossover(*p1, *p2, &mut thread_rng()).into_iter()
                }),
        );
        children.extend(parents.take(self.num_reserve).map(T::clone));
        for i in &mut children {
            if rng.gen_bool(self.mutation_probability) {
                i.mutate(&mut rng)
            }
            i.update_fitness();
        }
        sort_individuals(&mut children);
        self.individuals = children;
        Ok(())
    }

    pub fn get_best(&self) -> &T {
        self.individuals.last().unwrap()
    }

    pub fn get_worst(&self) -> &T {
        &self.individuals[0]
    }

    pub fn get_avg_fitness(&self) -> f64 {
        let sum: f64 = self.individuals.iter().map(|i| i.fitness()).sum();
        sum / self.individuals.len() as f64
    }
}

fn sort_individuals<T: Individual>(individuals: &mut Vec<T>) {
    individuals.sort_unstable_by(|x, y| x.fitness().partial_cmp(&y.fitness()).unwrap());
}
