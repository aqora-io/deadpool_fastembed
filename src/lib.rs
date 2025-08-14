use config::{Embedding, FastembedConfig};
use deadpool::managed;

mod config;

pub use config::Config;

/// [`Manager`] for creating and recycling fastembed.
///
/// [`Manager`]: managed::Manager
#[derive(Debug)]
pub struct Manager {
    config: FastembedConfig,
}

impl Manager {
    pub fn new(config: FastembedConfig) -> Self {
        Self { config }
    }
}

impl managed::Manager for Manager {
    type Type = Embedding;
    type Error = fastembed::Error;

    async fn create(&self) -> Result<Self::Type, Self::Error> {
        Embedding::try_new(self.config.clone())
    }

    async fn recycle(
        &self,
        _: &mut Self::Type,
        _: &managed::Metrics,
    ) -> managed::RecycleResult<Self::Error> {
        Ok(())
    }
}
