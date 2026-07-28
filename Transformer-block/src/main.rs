use rand::{Rng, RngExt};
use std::io;
use std::io::Write;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

const MANUAL_SEED: u64 = 42;

// Convention: 3D tensors are [batch, seq_len, dim]. 2D tensors are weights [rows, cols] (shared across batch).
struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
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

    // 2D indexing (for weight matrices: shape = [rows, cols])
    fn get(&self, row: usize, col: usize) -> f32 {
        let cols = self.shape[1];
        self.data[row * cols + col]
    }

    // 3D indexing (for batched tensors: shape = [batch, rows, cols])
    fn get3(&self, b: usize, row: usize, col: usize) -> f32 {
        let rows = self.shape[1];
        let cols = self.shape[2];
        self.data[b * rows * cols + row * cols + col]
    }

    // batched matmul: self [batch, m, k] · other [batch, k, n] -> [batch, m, n]
    fn matmul_batched(&self, other: &Tensor) -> Tensor {
        let batch = self.shape[0];
        let m = self.shape[1];
        let k = self.shape[2];
        let k2 = other.shape[1];
        let n = other.shape[2];
        assert_eq!(batch, other.shape[0], "batch size mismatch");
        assert_eq!(k, k2, "matmul_batched shape mismatch: {:?} vs {:?}", self.shape, other.shape);

        let mut data = vec![0.0; batch * m * n];
        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for kk in 0..k {
                        sum += self.get3(b, i, kk) * other.get3(b, kk, j);
                    }
                    data[b * m * n + i * n + j] = sum;
                }
            }
        }
        Tensor { data, shape: vec![batch, m, n] }
    }

    // broadcast matmul: self [batch, m, k] · weight [k, n] (2D, shared across batch) -> [batch, m, n]
    fn matmul_weight(&self, weight: &Tensor) -> Tensor {
        let batch = self.shape[0];
        let m = self.shape[1];
        let k = self.shape[2];
        let k2 = weight.shape[0];
        let n = weight.shape[1];
        assert_eq!(k, k2, "matmul_weight shape mismatch: {:?} vs {:?}", self.shape, weight.shape);

        let mut data = vec![0.0; batch * m * n];
        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for kk in 0..k {
                        sum += self.get3(b, i, kk) * weight.get(kk, j);
                    }
                    data[b * m * n + i * n + j] = sum;
                }
            }
        }
        Tensor { data, shape: vec![batch, m, n] }
    }

    // transpose last two dims, per batch: [batch, rows, cols] -> [batch, cols, rows]
    fn transpose_last(&self) -> Tensor {
        let batch = self.shape[0];
        let rows = self.shape[1];
        let cols = self.shape[2];
        let mut data = vec![0.0; batch * rows * cols];

        for b in 0..batch {
            for i in 0..rows {
                for j in 0..cols {
                    data[b * cols * rows + j * rows + i] = self.get3(b, i, j);
                }
            }
        }
        Tensor { data, shape: vec![batch, cols, rows] }
    }

    fn scalar_div(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x / scalar).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    // softmax over the last dimension, per batch, per row
    fn softmax_rows(&self) -> Tensor {
        let batch = self.shape[0];
        let rows = self.shape[1];
        let cols = self.shape[2];
        let mut data = vec![0.0; batch * rows * cols];

        for b in 0..batch {
            for i in 0..rows {
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..cols {
                    let v = self.get3(b, i, j);
                    if v > max_val { max_val = v; }
                }

                let mut row_exp = vec![0.0; cols];
                let mut sum = 0.0;
                for j in 0..cols {
                    let e = (self.get3(b, i, j) - max_val).exp();
                    row_exp[j] = e;
                    sum += e;
                }

                for j in 0..cols {
                    data[b * rows * cols + i * cols + j] = row_exp[j] / sum;
                }
            }
        }
        Tensor { data, shape: vec![batch, rows, cols] }
    }

    // layer norm over the last dimension, per batch, per row
    fn layer_norm(&self, epsilon: f32) -> Tensor {
        let batch = self.shape[0];
        let rows = self.shape[1];
        let cols = self.shape[2];
        let mut data = vec![0.0; batch * rows * cols];

        for b in 0..batch {
            for i in 0..rows {
                let mut sum = 0.0;
                for j in 0..cols {
                    sum += self.get3(b, i, j);
                }
                let mean = sum / cols as f32;

                let mut var_sum = 0.0;
                for j in 0..cols {
                    let diff = self.get3(b, i, j) - mean;
                    var_sum += diff * diff;
                }
                let variance = var_sum / cols as f32;
                let denom = (variance + epsilon).sqrt();

                for j in 0..cols {
                    data[b * rows * cols + i * cols + j] = (self.get3(b, i, j) - mean) / denom;
                }
            }
        }
        Tensor { data, shape: vec![batch, rows, cols] }
    }

    // adds a 1D bias vector to every row, broadcast across batch too
    fn add_bias(&self, bias: &Tensor) -> Tensor {
        let batch = self.shape[0];
        let rows = self.shape[1];
        let cols = self.shape[2];
        assert_eq!(cols, bias.data.len(), "bias length must match column count");

        let mut data = vec![0.0; batch * rows * cols];
        for b in 0..batch {
            for i in 0..rows {
                for j in 0..cols {
                    data[b * rows * cols + i * cols + j] = self.get3(b, i, j) + bias.data[j];
                }
            }
        }
        Tensor { data, shape: vec![batch, rows, cols] }
    }

    fn gelu(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| {
            0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        }).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    // element-wise add, requires identical shapes
    fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "add shape mismatch: {:?} vs {:?}", self.shape, other.shape);
        let data: Vec<f32> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    // dropout: zero out each element with probability p, rescale survivors by 1/(1-p)
    fn dropout(&self, p: f32, rng: &mut StdRng) -> Tensor {
        if p <= 0.0 {
            return Tensor { data: self.data.clone(), shape: self.shape.clone() };
        }
        let scale = 1.0 / (1.0 - p);
        let data: Vec<f32> = self.data.iter().map(|&x| {
            if rng.random::<f32>() < p { 0.0 } else { x * scale }
        }).collect();
        Tensor { data, shape: self.shape.clone() }
    }

    // stacks a batch of sequences: each sequence is a Vec<&Tensor> of [d_model] vectors -> [batch, seq_len, d_model]
    fn stack_batch(sequences: &[Vec<&Tensor>]) -> Tensor {
        let batch = sequences.len();
        let seq_len = sequences[0].len();
        let d_model = sequences[0][0].data.len();

        let mut data = Vec::with_capacity(batch * seq_len * d_model);
        for seq in sequences {
            assert_eq!(seq.len(), seq_len, "all sequences in a batch must have the same length");
            for token_vec in seq {
                assert_eq!(token_vec.data.len(), d_model, "all token vectors must have same length");
                data.extend_from_slice(&token_vec.data);
            }
        }
        Tensor { data, shape: vec![batch, seq_len, d_model] }
    }
}

fn read_shape() -> Vec<usize> {
    print!("Enter shape (space-separated, e.g. 2 4 8): ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");
    input.trim().split_whitespace().map(|s| s.parse::<usize>().expect("Not a valid number")).collect()
}

fn read_text(prompt: &str) -> String {
    print!("{prompt}");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");
    input.trim().to_string()
}

fn read_usize(prompt: &str) -> usize {
    print!("{prompt}");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");
    input.trim().parse::<usize>().expect("Not a valid number")
}

fn read_f32(prompt: &str) -> f32 {
    print!("{prompt}");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");
    input.trim().parse::<f32>().expect("Not a valid number")
}

struct Tokenizer {
    vocab: HashMap<String, usize>,
    next_id: usize,
}

impl Tokenizer {
    fn new() -> Self {
        let mut vocab = HashMap::new();
        vocab.insert("<PAD>".to_string(), 0);
        Tokenizer { vocab, next_id: 1 } // real words start at id 1
    }

    fn tokenize(&mut self, text: &str) -> Vec<usize> {
        text.split_whitespace().map(|word| self.get_or_create_id(word)).collect()
    }

    fn get_or_create_id(&mut self, word: &str) -> usize {
        if let Some(&id) = self.vocab.get(word) {
            id
        } else {
            let id = self.next_id;
            self.vocab.insert(word.to_string(), id);
            self.next_id += 1;
            id
        }
    }
}

struct Embedding {
    table: Vec<Tensor>,
}

impl Embedding {
    fn new(vocab_size: usize, dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(MANUAL_SEED);
        let total = vocab_size * dim;
        let flat_data: Vec<f32> = (0..total).map(|_| rng.random_range(-1.0..1.0)).collect();
        let table: Vec<Tensor> = flat_data.chunks(dim)
        .map(|chunk: &[f32]| Tensor { data: chunk.to_vec(), shape: vec![dim] }).collect();
        Embedding { table }
    }

    fn embed(&self, token_ids: &[usize]) -> Vec<&Tensor> {
        token_ids.iter().map(|&id| &self.table[id]).collect()
    }
}

struct AttentionWeights {
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
}

struct MlpWeights {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
}

struct TransformerLayer {
    attention: AttentionWeights,
    mlp: MlpWeights,
}

impl TransformerLayer {
    fn new(d_model: usize, d_k: usize, d_v: usize, d_ff: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let make_matrix = |rng: &mut StdRng, rows: usize, cols: usize| -> Tensor {
            let data: Vec<f32> = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect();
            Tensor { data, shape: vec![rows, cols] }
        };
        let make_vector = |rng: &mut StdRng, len: usize| -> Tensor {
            let data: Vec<f32> = (0..len).map(|_| rng.random_range(-1.0..1.0)).collect();
            Tensor { data, shape: vec![len] }
        };

        let attention = AttentionWeights {
            w_q: make_matrix(&mut rng, d_model, d_k),
            w_k: make_matrix(&mut rng, d_model, d_k),
            w_v: make_matrix(&mut rng, d_model, d_v),
        };
        let mlp = MlpWeights {
            w1: make_matrix(&mut rng, d_model, d_ff),
            b1: make_vector(&mut rng, d_ff),
            w2: make_matrix(&mut rng, d_ff, d_model),
            b2: make_vector(&mut rng, d_model),
        };

        TransformerLayer { attention, mlp }
    }

    // x: [batch, seq_len, d_model]
    fn attention(&self, x: &Tensor, dropout_p: f32, rng: &mut StdRng) -> Tensor {
        let q = x.matmul_weight(&self.attention.w_q);
        let k = x.matmul_weight(&self.attention.w_k);
        let v = x.matmul_weight(&self.attention.w_v);

        let d_k = self.attention.w_q.shape[1] as f32;
        let scores = q.matmul_batched(&k.transpose_last()).scalar_div(d_k.sqrt());
        let weights = scores.softmax_rows().dropout(dropout_p, rng);

        weights.matmul_batched(&v)
    }

    fn mlp(&self, x: &Tensor, dropout_p: f32, rng: &mut StdRng) -> Tensor {
        let hidden = x.matmul_weight(&self.mlp.w1).add_bias(&self.mlp.b1).gelu().dropout(dropout_p, rng);
        hidden.matmul_weight(&self.mlp.w2).add_bias(&self.mlp.b2)
    }

    fn forward(&self, x: &Tensor, dropout_p: f32, rng: &mut StdRng) -> Tensor {
        let epsilon = 1e-5;

        let attn_out = self.attention(x, dropout_p, rng);
        let x1 = x.add(&attn_out).layer_norm(epsilon);

        let mlp_out = self.mlp(&x1, dropout_p, rng);
        x1.add(&mlp_out).layer_norm(epsilon)
    }
}

fn main() {
    let batch_size = read_usize("Enter batch size (number of sentences): ");

    let mut tokenizer = Tokenizer::new();
    let mut all_token_ids: Vec<Vec<usize>> = Vec::new();

    for i in 0..batch_size {
        let text = read_text(&format!("Enter text for sequence {}: ", i + 1));
        let ids = tokenizer.tokenize(&text);
        all_token_ids.push(ids);
    }

    let seq_len = all_token_ids.iter().map(|ids| ids.len()).max().unwrap();

    for ids in all_token_ids.iter_mut() {
        while ids.len() < seq_len {
            ids.push(0); // pad with <PAD> token id
        }
    }

    let shape = read_shape();
    let d_model: usize = shape.iter().product();

    let embedding = Embedding::new(tokenizer.next_id, d_model);

    let sequences: Vec<Vec<&Tensor>> = all_token_ids.iter()
        .map(|ids| embedding.embed(ids))
        .collect();
    let x = Tensor::stack_batch(&sequences); // [batch, seq_len, d_model]

    let num_heads = read_usize("Enter number of attention heads: ");
    let seq_len = all_token_ids[0].len();
    for ids in &all_token_ids {
        assert_eq!(ids.len(), seq_len, "...");
    }
    let d_k = d_model / num_heads;
    let d_v = d_model;
    let d_ff = read_usize("Enter feed-forward hidden size (d_ff): ");
    let num_layers = read_usize("Enter number of layers: ");
    let dropout_p = read_f32("Enter dropout probability (e.g. 0.1, or 0 to disable): ");

    let mut rng = StdRng::seed_from_u64(MANUAL_SEED + 1000);

    let mut output = x;
    for i in 0..num_layers {
        let layer = TransformerLayer::new(d_model, d_k, d_v, d_ff, MANUAL_SEED + i as u64);
        output = layer.forward(&output, dropout_p, &mut rng);
    }

    println!("{}", output.to_nested_string());
}