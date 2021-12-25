use anyhow::*;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::{collections::HashSet, iter::repeat, path::Path};

use crate::ga::*;
use crate::tsp_data::Tsp;
use crate::{Opt, OutputContent};

use rand::{
    distributions::Uniform,
    prelude::{Distribution, SliceRandom},
    thread_rng, Rng,
};
use rayon::slice::ParallelSlice;

#[derive(Clone)]
struct Spec<'a, T: Tsp> {
    tsp: &'a T,
    num_crossover_points: usize,
    num_mutation_points: usize,
}
struct Route<'a, T: Tsp> {
    route: Vec<usize>,
    fitness: f64,
    spec: &'a Spec<'a, T>,
}

impl<'a, T: Tsp> Clone for Route<'a, T> {
    fn clone(&self) -> Self {
        Self {
            route: self.route.clone(),
            fitness: self.fitness,
            spec: self.spec,
        }
    }
}

impl<'a, T: Tsp> Individual for Route<'a, T> {
    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn mutate<R: Rng>(&mut self, mut rng: &mut R) {
        let range: Uniform<usize> = Uniform::from(1..self.route.len() - 1);
        let mut mutation_indexes: HashSet<usize> =
            HashSet::with_capacity(self.spec.num_mutation_points);
        mutation_indexes.extend(
            range
                .sample_iter(&mut rng)
                .take(self.spec.num_mutation_points),
        );
        while mutation_indexes.len() != self.spec.num_mutation_points {
            mutation_indexes.insert(range.sample(rng));
        }
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

    fn crossover<R: Rng>(p1: &Self, p2: &Self, rng: &mut R) -> [Self; 2] {
        let range: Uniform<usize> = Uniform::from(1..p1.route.len());
        let mut crossover_points = Vec::with_capacity(p1.spec.num_crossover_points + 2);
        crossover_points.extend(range.sample_iter(rng).take(p1.spec.num_crossover_points));
        crossover_points.push(1);
        crossover_points.push(p1.spec.tsp.dim());
        crossover_points.sort_unstable();
        [
            get_one_child::<_, 0, 1>(&crossover_points, p1, p2),
            get_one_child::<_, 1, 0>(&crossover_points, p1, p2),
        ]
    }

    fn update_fitness(&mut self) {
        self.fitness = 1f64 / total_weight(&self.route, self.spec.tsp)
    }
}

fn get_one_child<'a, T: Tsp, const N: usize, const M: usize>(
    crossover_points: &[usize],
    p1: &Route<'a, T>,
    p2: &Route<T>,
) -> Route<'a, T> {
    let segments = crossover_points.windows(2);
    let gene_from_p1: HashSet<usize> = segments
        .clone()
        .skip(N)
        .step_by(2)
        .flat_map(|window| &p1.route[window[0]..window[1]])
        .copied()
        .collect();
    let mut child = p1.clone();
    let mut gene_from_p2 = p2.route[1..p2.route.len() - 1]
        .iter()
        .filter(|&i| !gene_from_p1.contains(i))
        .copied();
    segments.skip(M).step_by(2).for_each(|window| {
        for i in &mut child.route[window[0]..window[1]] {
            *i = gene_from_p2.next().unwrap();
        }
    });
    child
}

fn total_weight<T: Tsp>(route: &[usize], tsp: &T) -> f64 {
    route
        .par_windows(2)
        .with_min_len(256)
        .map(|edge| /*get_weight(tsp, edge[0], edge[1])*/tsp.weight(edge[0], edge[1]))
        .sum()
}

pub fn solve_tsp<T: Tsp>(opt: Opt, tsp: T) -> Result<()> {
    let spec = Spec {
        tsp: &tsp,
        num_crossover_points: opt.num_crossover_point,
        num_mutation_points: opt.num_mutation_points,
    };
    let mut init_route: Vec<usize> = Vec::with_capacity(tsp.dim() + 2);
    init_route.extend(0..tsp.dim());
    init_route.push(0);

    let mut rng = thread_rng();
    let individuals = repeat(init_route.clone())
        .take(opt.population)
        .map(|mut i| {
            i[1..tsp.dim()].shuffle(&mut rng);
            Route {
                fitness: 1f64 / total_weight(&i, &tsp),
                spec: &spec,
                route: i,
            }
        })
        .collect();

    let mut population = Population::new(
        individuals,
        opt.population,
        opt.crossover_probability,
        opt.mutation_probability,
    );
    let bar = ProgressBar::new(opt.max_generation as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed}|{eta}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}"),
    );
    match opt.expected_answer {
        Some(ans) => {
            run_with_expected_fitness(1f64 / ans, opt.max_generation, &mut population, bar)?
        }
        None => {
            match &opt.output_content {
                Some(content) => run_with_output(
                    *content,
                    &opt.output_file.unwrap(),
                    &mut population,
                    opt.max_generation,
                    bar,
                )?,
                None => run(opt.max_generation, &mut population, bar),
            }
            let ans = total_weight(population.get_best().route.as_ref(), &tsp);
            println!("{}", ans);
        }
    }
    Ok(())
}

fn run_with_output<T: Tsp>(
    content: OutputContent,
    output: &Path,
    population: &mut Population<Route<T>>,
    generation: usize,
    bar: ProgressBar,
) -> Result<()> {
    match content {
        OutputContent::Statistics => {
            run_with_statistics_output(output, population, generation, bar)
        }
        OutputContent::Routes => run_with_routes_output(output, population, generation, bar),
    }
}

fn run_with_routes_output<T: Tsp>(
    output: &Path,
    population: &mut Population<Route<T>>,
    generation: usize,
    bar: ProgressBar,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output)?;
    for _ in 0..generation {
        population.evolve().unwrap();
        bar.inc(1);
        let route = best_route(population);
        writer.serialize(route)?;
    }
    Ok(())
}

fn best_route<T: Tsp>(population: &Population<Route<T>>) -> Vec<usize> {
    let best = population.get_best();
    best.route.clone()
}

fn run_with_expected_fitness<T: Tsp>(
    fitness: f64,
    max_generation: usize,
    population: &mut Population<Route<T>>,
    bar: ProgressBar,
) -> Result<()> {
    let base = fitness - population.get_best().fitness;
    bar.set_length(100);
    for generation in 1..=max_generation {
        population.evolve().unwrap();
        let current_best = population.get_best().fitness;
        if current_best >= fitness {
            bar.abandon();
            println!("{}", generation);
            return Ok(());
        }
        bar.set_position(100 - ((fitness - current_best) / base * 100.) as u64);
    }
    Err(anyhow!(
        "Generation limit reached. Current answer: {}.",
        1f64 / population.get_best().fitness
    ))
}

fn run_with_statistics_output<T: Tsp>(
    output: &Path,
    population: &mut Population<Route<T>>,
    generation: usize,
    bar: ProgressBar,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output)?;
    for _ in 0..generation {
        population.evolve().unwrap();
        bar.inc(1);
        let max = population.get_best().fitness;
        let avg = population.get_avg_fitness();
        let min = population.get_worst().fitness;
        writer.serialize(&[max, avg, min])?;
    }
    Ok(())
}

fn run<T: Tsp>(generation: usize, population: &mut Population<Route<T>>, bar: ProgressBar) {
    for _ in 0..generation {
        population.evolve().unwrap();
        bar.inc(1);
    }
}
