use rand::{Rng, RngExt};
use std::io;
use std::io::Write;
use rand::SeedableRng;
use rand::rngs::StdRng;

const MANUAL_SEED: u64 = 42;

struct Tensor{
    data: Vec<f32>,
    shape: Vec<usize>
}

impl Tensor {
    fn to_nested_string(&self) -> String {
        Self::format_dim(&self.data, &self.shape)
    }

    fn format_dim(data: &[f32], shape: &[usize]) -> String {
        if shape.is_empty() {
            return format!("{:.3}", data[0]);
        }
        if shape.len() == 1 {
            let items: Vec<String> = data.iter().map(|x| format!("{:.3}", x)).collect();
            return format!("[{}]", items.join(", "));
        }
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

fn read_text() -> String {
    print!("Enter text to tokenize: ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");

    input.trim().to_string()
}

use std::collections::HashMap;

struct Tokenizer {
    vocab: HashMap<String, usize>, // word -> id
    next_id: usize,
}

impl Tokenizer {
    fn new() -> Self {
        Tokenizer { vocab: HashMap::new(), next_id: 0 }
    }

    // splits text into words, assigns each unique word an id
    fn tokenize(&mut self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| self.get_or_create_id(word))
            .collect()
    }

    fn get_or_create_id(&mut self, word: &str) -> usize {
        if let Some(&id) = self.vocab.get(word) {
            id // already seen this word, reuse its id
        } else {
            let id = self.next_id;
            self.vocab.insert(word.to_string(), id);
            self.next_id += 1;
            id
        }
    }
}

struct Embedding {
    table: Vec<Tensor>, // table[id] = vector for that token id
    shape: Vec<usize>,
}

impl Embedding {
    fn new(vocab_size: usize, shape: Vec<usize>) -> Self {
        let mut rng = StdRng::seed_from_u64(MANUAL_SEED); // any fixed number works, MANUAL_SEED is arbitrary

        let per_token: usize = shape.iter().product(); // floats needed per token
        let total: usize = vocab_size * per_token;      // floats needed for whole table

        // one flat generation pass, same style as Tensor::random
        let flat_data: Vec<f32> = (0..total).map(|_| rng.random_range(-1.0..1.0)).collect();

        // chunk the flat buffer into `vocab_size` pieces of `per_token` floats each,
        // wrapping each piece into its own Tensor with the given shape
        let table: Vec<Tensor> = flat_data
            .chunks(per_token)
            .map(|chunk| Tensor { data: chunk.to_vec(), shape: shape.clone() })
            .collect();

        Embedding { table, shape }
    }

    fn embed(&self, token_ids: &[usize]) -> Vec<&Tensor> {
        token_ids.iter().map(|&id| &self.table[id]).collect()
    }
}

fn main() {
    let mut tokenizer = Tokenizer::new();
    let text = read_text();
    let token_ids = tokenizer.tokenize(&text);

    println!("Token IDs: {:?}", token_ids);
    println!("Vocab: {:?}", tokenizer.vocab);

    let shape = read_shape();
    println!("{:?}", &shape);
    let embedding = Embedding::new(tokenizer.next_id, shape);
    let vectors = embedding.embed(&token_ids);

    for (word_id, vec) in token_ids.iter().zip(vectors.iter()) {
    println!("id {} -> {}", word_id, vec.to_nested_string());
    }
}