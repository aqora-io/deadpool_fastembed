use std::path::PathBuf;

use deadpool::{
    Runtime,
    managed::{CreatePoolError, Pool, PoolBuilder, PoolConfig},
};
use fastembed::{
    EmbeddingModel, ExecutionProviderDispatch, ImageEmbedding, ImageEmbeddingModel,
    ImageInitOptions, InitOptions, RerankInitOptions, RerankerModel, SparseInitOptions,
    SparseModel, SparseTextEmbedding, TextEmbedding, TextRerank,
};

#[derive(Debug, Clone)]
pub enum ModelKind {
    Text(EmbeddingModel),
    Image(ImageEmbeddingModel),
    Sparse(SparseModel),
    ReRanking(RerankerModel),
}

impl Default for ModelKind {
    fn default() -> Self {
        Self::Text(EmbeddingModel::BGESmallENV15)
    }
}

#[derive(Debug, Clone, Default)]
pub struct FastembedConfig {
    pub model: ModelKind,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
}

impl FastembedConfig {
    pub fn new(model: ModelKind) -> Self {
        Self {
            model,
            show_download_progress: false,
            ..Default::default()
        }
    }
}

pub enum Embedding {
    Text(TextEmbedding),
    Image(ImageEmbedding),
    Sparse(SparseTextEmbedding),
    ReRanking(TextRerank),
}

impl Embedding {
    /// from a given [`FastembedConfig`] init the right embedding defined in fastembed
    pub fn try_new(config: FastembedConfig) -> Result<Self, fastembed::Error> {
        Ok(match config.model {
            ModelKind::Text(model) => Self::Text(TextEmbedding::try_new(
                InitOptions::new(model)
                    .with_max_length(config.max_length)
                    .with_cache_dir(config.cache_dir)
                    .with_execution_providers(config.execution_providers)
                    .with_show_download_progress(config.show_download_progress),
            )?),
            ModelKind::Image(embedding) => Self::Image(ImageEmbedding::try_new(
                ImageInitOptions::new(embedding)
                    .with_cache_dir(config.cache_dir.clone())
                    .with_execution_providers(config.execution_providers)
                    .with_show_download_progress(config.show_download_progress),
            )?),
            ModelKind::Sparse(embedding) => Self::Sparse(SparseTextEmbedding::try_new(
                SparseInitOptions::new(embedding)
                    .with_max_length(config.max_length)
                    .with_cache_dir(config.cache_dir)
                    .with_execution_providers(config.execution_providers)
                    .with_show_download_progress(config.show_download_progress),
            )?),
            ModelKind::ReRanking(embedding) => Self::ReRanking(TextRerank::try_new(
                RerankInitOptions::new(embedding)
                    .with_max_length(config.max_length)
                    .with_cache_dir(config.cache_dir.clone())
                    .with_execution_providers(config.execution_providers)
                    .with_show_download_progress(config.show_download_progress),
            )?),
        })
    }
}

/// Configuration object.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(crate = "serde"))]
pub struct Config {
    /// All fastembed options when init a embedding
    pub fastembed_config: FastembedConfig,

    /// [`Pool`] configuration.
    pub pool: Option<PoolConfig>,
}

impl Config {
    pub fn create_pool(
        &self,
        runtime: Option<Runtime>,
    ) -> Result<Pool<crate::Manager>, CreatePoolError<ConfigError>> {
        let mut builder = self.builder().map_err(CreatePoolError::Config)?;
        if let Some(runtime) = runtime {
            builder = builder.runtime(runtime);
        }
        builder.build().map_err(CreatePoolError::Build)
    }

    pub fn builder(&self) -> Result<PoolBuilder<crate::Manager>, ConfigError> {
        let manager = crate::Manager::new(self.fastembed_config.clone());
        Ok(Pool::builder(manager).config(self.get_pool_config()))
    }

    #[must_use]
    pub fn get_pool_config(&self) -> PoolConfig {
        self.pool.unwrap_or_default()
    }

    #[must_use]
    pub fn from_model(model: impl Into<ModelKind>) -> Self {
        Self {
            fastembed_config: FastembedConfig::new(model.into()),
            pool: None,
        }
    }

    #[must_use]
    pub fn from_config(config: impl Into<FastembedConfig>) -> Self {
        Self {
            fastembed_config: config.into(),
            pool: None,
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    NotFound,
    Fastembed(fastembed::Error),
}

impl From<fastembed::Error> for ConfigError {
    fn from(value: fastembed::Error) -> Self {
        Self::Fastembed(value)
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "model not foud"),
            Self::Fastembed(e) => write!(f, "fastembed: {e}"),
        }
    }
}

impl std::error::Error for ConfigError {}
