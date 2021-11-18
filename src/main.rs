use std::path::Path;

mod ga;
mod tsp;

fn main() {
    let path = Path::new("/home/mike/repos/rust/genetictsp/fnl4461.tsp");
    tsp::solve_tsp(path);
}
