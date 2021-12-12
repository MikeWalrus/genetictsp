use std::path::PathBuf;

use anyhow::*;
use structopt::{clap::arg_enum, StructOpt};
use tsp_data::{Euc2d, Matrix};

mod ga;
mod tsp;
mod tsp_data;

arg_enum! {
    #[derive(Clone, Copy)]
    enum OutputContent {
        Statistics,
        Routes
    }
}

#[derive(StructOpt)]
pub struct Opt {
    #[structopt(
        name = "output_content", long = "output_content", short = "O",
        possible_values = &OutputContent::variants(),
        requires = "output"
    )]
    output_content: Option<OutputContent>,
    #[structopt(
        name = "expected_answer",
        long = "expected",
        short = "e",
        conflicts_with = "output_content",
        conflicts_with = "output"
    )]
    expected_answer: Option<f64>,
    #[structopt(long = "tsp_file", short = "i", parse(from_os_str))]
    tsp_file_path: PathBuf,
    #[structopt(
        name = "output",
        long = "output",
        short = "o",
        parse(from_os_str),
        requires = "output_content"
    )]
    output_file: Option<PathBuf>,
    #[structopt(short = "g", default_value = "1000")]
    max_generation: usize,
    #[structopt(short = "c", default_value = "3")]
    num_crossover_point: usize,
    #[structopt(short = "m", default_value = "3")]
    num_mutation_points: usize,
    #[structopt(short = "C", default_value = "0.5")]
    crossover_probability: f64,
    #[structopt(short = "M", default_value = "0.1")]
    mutation_probability: f64,
    #[structopt(short = "p", default_value = "100")]
    population: usize,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();
    let instance =
        tspf::TspBuilder::parse_path(&opt.tsp_file_path).map_err(|e| anyhow!(e.to_string()))?;
    let dim = instance.dim();

    match instance.weight_format() {
        tspf::WeightFormat::Function | tspf::WeightFormat::Undefined => {
            match instance.coord_kind() {
                tspf::CoordKind::Coord2d => match instance.weight_kind() {
                    tspf::WeightKind::Euc2d => {
                        let tsp = Euc2d::new(instance.node_coords(), dim);
                        tsp::solve_tsp(opt, tsp)
                    }
                    _ => unimplemented!(),
                },
                _ => unimplemented!(),
            }
        }
        tspf::WeightFormat::LowerDiagRow => {
            let tsp = Matrix::new_lower_diag_row(instance.edge_weights(), dim);
            tsp::solve_tsp(opt, tsp)
        }
        _ => unimplemented!(),
    }
}
