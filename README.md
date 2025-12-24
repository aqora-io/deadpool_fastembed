# Deadpool for [FastEmbed-rs](https://crates.io/crates/fastembed)

Deadpool is a dead simple async pool for connections and objects of any type.

This crate implements a [`deadpool`](https://crates.io/crates/deadpool) manager for [`fastembed`](https://crates.io/crates/fastembed).

## Features

| Feature                 | Description                                                                 | Extra dependencies                | Default |
| ----------------------- | --------------------------------------------------------------------------- | --------------------------------- | ------- |
| `rt_tokio_1`            | Enable support for [tokio](https://crates.io/crates/tokio) runtime          | `deadpool/rt_tokio_1`             | yes     |
| `rt_async-std_1`        | Enable support for [async-std](https://crates.io/crates/async-std) runtime  | `deadpool/rt_async-std_1`         | no      |
| `hf-hub`                | Enable support for loading models from Hugging Face Hub                     | `fastembed/hf-hub`                | no      |
| `hf-hub-native-tls`     | Hugging Face Hub with [native-tls](https://crates.io/crates/native-tls)     | `fastembed/hf-hub-native-tls`     | no      |
| `hf-hub-rustls-tls`     | Hugging Face Hub with [rustls](https://crates.io/crates/rustls) TLS backend | `fastembed/hf-hub-rustls-tls`     | no      |
| `image-models`          | Fastembed with image models enabled                                         | `fastembed/image-models`          | no      |
| `ort-download-binaries` | Automatically download ONNX Runtime binaries                                | `fastembed/ort-download-binaries` | no      |
| `ort-load-dynamic`      | Dynamically load ONNX Runtime library at runtime                            | `fastembed/ort-load-dynamic`      | no      |
| `optimum-cli`           | Enable compatibility with Hugging Face Optimum CLI                          | `fastembed/optimum-cli`           | no      |
| `online`                | Allow fetching models and resources at runtime                              | `fastembed/online`                | no      |

All of the features of [fastembed](https://crates.io/crates/fastembed) are also re-exported.
For example, enabling the feature `hf-hub-rustls-tls` here will also enable `hf-hub-rustls-tls` in the `fastembed` crate.

## Example

```rust
use deadpool_fastembed::{Config, ModelKind, Pool};
use fastembed::{EmbeddingModel, TextEmbedding};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::from_model(ModelKind::text(EmbeddingModel::AllMiniLML6V2));

    let pool: Pool = config.create_pool(None)?;

    let mut embedding = pool.get().await?;

    if let Some(text_embed) = embedding.as_text_mut() {
        let vectors = text_embed.embed(vec!["hello world".to_string()]).await?;
        println!("Embedding length: {}", vectors[0].len());
    }

    Ok(())
}

```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
