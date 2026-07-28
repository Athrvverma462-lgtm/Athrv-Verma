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

     // 2D indexing helper: converts (row, col) into the flat buffer index
    fn get(&self, row: usize, col: usize) -> f32 {
        let cols = self.shape[1];
        self.data[row * cols + col]
    }

    fn matmul(&self, other: &Tensor) -> Tensor {
        // self: [rows_a, cols_a], other: [rows_b, cols_b], require cols_a == rows_b
        let rows_a = self.shape[0];
        let cols_a = self.shape[1];
        let rows_b = other.shape[0];
        let cols_b = other.shape[1];

        assert_eq!(cols_a, rows_b, "matmul shape mismatch: {:?} vs {:?}", self.shape, other.shape);

        let mut data = vec![0.0; rows_a * cols_b];

        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0;
                for k in 0..cols_a {
                    sum += self.get(i, k) * other.get(k, j);
                }
                data[i * cols_b + j] = sum;
            }
        }

        Tensor { data, shape: vec![rows_a, cols_b] }
    }

    fn transpose(&self) -> Tensor {
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                // swap positions: element at (i,j) goes to (j,i) in the new layout
                data[j * rows + i] = self.get(i, j);
            }
        }

        Tensor { data, shape: vec![cols, rows] } // shape dimensions swapped too
    }

    fn scalar_div(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x / scalar).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    fn softmax_rows(&self) -> Tensor {
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = vec![0.0; rows * cols];

        for i in 0..rows {
            // 1. find the max value in this row (for numerical stability)
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..cols {
                let v = self.get(i, j);
                if v > max_val {
                    max_val = v;
                }
            }

            // 2. exponentiate each element (shifted by max_val to avoid overflow)
            let mut row_exp = vec![0.0; cols];
            let mut sum = 0.0;
            for j in 0..cols {
                let e = (self.get(i, j) - max_val).exp();
                row_exp[j] = e;
                sum += e;
            }

            // 3. divide each exponentiated value by the row's sum
            for j in 0..cols {
                data[i * cols + j] = row_exp[j] / sum;
            }
        }

        Tensor { data, shape: vec![rows, cols] }
    }

    fn layer_norm(&self, epsilon: f32) -> Tensor {
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = vec![0.0; rows * cols];

        for i in 0..rows {
            // 1. compute mean of this row
            let mut sum = 0.0;
            for j in 0..cols {
                sum += self.get(i, j);
            }
            let mean = sum / cols as f32;

            // 2. compute variance of this row
            let mut var_sum = 0.0;
            for j in 0..cols {
                let diff = self.get(i, j) - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / cols as f32;

            // 3. normalize: (x - mean) / sqrt(variance + epsilon)
            let denom = (variance + epsilon).sqrt();
            for j in 0..cols {
                data[i * cols + j] = (self.get(i, j) - mean) / denom;
            }
        }

        Tensor { data, shape: vec![rows, cols] }
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

struct AttentionWeights {
    w_q: Tensor, // [d_model, d_k]
    w_k: Tensor, // [d_model, d_k]
    w_v: Tensor, // [d_model, d_v]
}

struct MlpWeights {
    w1: Tensor, // [d_model, d_ff]
    w2: Tensor, // [d_ff, d_model]
}

struct TransformerLayer {
    attention: AttentionWeights,
    mlp: MlpWeights,
}

impl TransformerLayer {
    // one seed per layer so layers don't accidentally share identical weights
    fn new(d_model: usize, d_k: usize, d_v: usize, d_ff: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let make_matrix = |rng: &mut StdRng, rows: usize, cols: usize| -> Tensor {
            let total = rows * cols;
            let data: Vec<f32> = (0..total).map(|_| rng.random_range(-1.0..1.0)).collect();
            Tensor { data, shape: vec![rows, cols] }
        };

        let attention = AttentionWeights {
            w_q: make_matrix(&mut rng, d_model, d_k),
            w_k: make_matrix(&mut rng, d_model, d_k),
            w_v: make_matrix(&mut rng, d_model, d_v),
        };

        let mlp = MlpWeights {
            w1: make_matrix(&mut rng, d_model, d_ff),
            w2: make_matrix(&mut rng, d_ff, d_model),
        };

        TransformerLayer { attention, mlp }
    }

    // stub for now — will compute Q = X·W_Q, K = X·W_K, V = X·W_V, then scaled dot-product attention
    fn attention(&self, x: &Tensor) -> Tensor {
        let q = x.matmul(&self.attention.w_q);
        let k = x.matmul(&self.attention.w_k);
        let v = x.matmul(&self.attention.w_v);

        let d_k = self.attention.w_q.shape[1] as f32;
        let scores = q.matmul(&k.transpose()).scalar_div(d_k.sqrt());
        let weights = scores.softmax_rows();

        weights.matmul(&v)
    }

    // stub for now — will compute a simple feed-forward: relu(X·W1)·W2
    fn mlp(&self, x: &Tensor) -> Tensor {
        todo!("matmul with w1, activation, matmul with w2")
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let epsilon = 1e-5;

        let attn_out = self.attention(x);
        let mut x1 = x.add(&attn_out);
        x1 = x1.layer_norm(epsilon);

        let mlp_out = self.mlp(&x1);
        let mut x2 = x1.add(&mlp_out);
        x2 = x2.layer_norm(epsilon);

        x2 // returned — no semicolon
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