use rand::{Rng, RngExt};
use std::io;
use std::io::Write;

struct Tensor{
    data: Vec<f32>,
    shape: Vec<usize>
}

impl Tensor {
    fn random(shape: Vec<usize>) -> Self {
        let mut rng = rand::rng();
        let total: usize = shape.iter().product();
        let data = (0..total).map(|_| rng.random_range(-1.0..1.0)).collect();
        Tensor { data, shape }
    }

    // recursively formats `data` according to `shape`, one dimension at a time
    fn to_nested_string(&self) -> String {
        Self::format_dim(&self.data, &self.shape)
    }

    fn format_dim(data: &[f32], shape: &[usize]) -> String {
        // base case: no dimensions left, we're at a single scalar
        if shape.is_empty() {
            return format!("{:.3}", data[0]);
        }

        // base case: last dimension, print a flat row like [0.1, -0.2, 0.5]
        if shape.len() == 1 {
            let items: Vec<String> = data.iter().map(|x| format!("{:.3}", x)).collect();
            return format!("[{}]", items.join(", "));
        }

        // recursive case: split `data` into `shape[0]` chunks,
        // each chunk handled by the remaining shape[1..]
        let chunk_size: usize = shape[1..].iter().product();
        let parts: Vec<String> = data
            .chunks(chunk_size)
            .map(|chunk| Self::format_dim(chunk, &shape[1..]))
            .collect();

        format!("[{}]", parts.join(",\n"))
    }
}

fn read_shape() -> Vec<usize> {
    print!("Enter shape (space-separated, e.g. 2 4 8): ");
    io::stdout().flush().unwrap(); // ensures prompt prints before input

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");

    input
        .trim()                     // remove trailing newline
        .split_whitespace()         // split on spaces
        .map(|s| s.parse::<usize>().expect("Not a valid number"))
        .collect()
}

fn main() {
    let shape = read_shape();
    println!("Shape entered: {:?}", shape);

    let t = Tensor::random(shape);
    println!("{}", t.to_nested_string());
}