use std::path::Path;

mod ga;
mod tsp;

fn main() {
    let path = Path::new("/home/mike/repos/rust/genetictsp/gr17.tsp");
    tsp::solve_tsp(path);
}
