use crate::{CreatePoolError, Pool, PoolBuilder, PoolConfig, Runtime};
use fastembed::{
    EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, RerankInitOptions,
    RerankerModel, SparseInitOptions, SparseModel, SparseTextEmbedding, TextEmbedding,
    TextInitOptions, TextRerank,
};

#[derive(Debug, Clone)]
pub enum ModelKind {
    Text(TextInitOptions),
    Image(ImageInitOptions),
    Sparse(SparseInitOptions),
    Rerank(RerankInitOptions),
}

impl Default for ModelKind {
    fn default() -> Self {
        Self::text(EmbeddingModel::default())
    }
}

impl ModelKind {
    pub fn text(model: EmbeddingModel) -> Self {
        Self::Text(TextInitOptions::new(model))
    }

    pub fn image(model: ImageEmbeddingModel) -> Self {
        Self::Image(ImageInitOptions::new(model))
    }

    pub fn sparse(model: SparseModel) -> Self {
        Self::Sparse(SparseInitOptions::new(model))
    }

    pub fn rerank(model: RerankerModel) -> Self {
        Self::Rerank(RerankInitOptions::new(model))
    }
}

pub enum EmbeddingKind {
    Text(TextEmbedding),
    Image(ImageEmbedding),
    Sparse(SparseTextEmbedding),
    ReRanking(TextRerank),
}

impl EmbeddingKind {
    pub fn try_new(model: &ModelKind) -> Result<Self, fastembed::Error> {
        Ok(match model.to_owned() {
            ModelKind::Text(opts) => Self::Text(TextEmbedding::try_new(opts)?),
            ModelKind::Image(opts) => Self::Image(ImageEmbedding::try_new(opts)?),
            ModelKind::Sparse(opts) => Self::Sparse(SparseTextEmbedding::try_new(opts)?),
            ModelKind::Rerank(opts) => Self::ReRanking(TextRerank::try_new(opts)?),
        })
    }
}

/// Configuration object.
#[derive(Clone, Debug)]
pub struct Config {
    pub model: ModelKind,

    /// [`Pool`] configuration.
    pub pool: Option<PoolConfig>,
}

impl Config {
    pub fn create_pool(&self, runtime: Option<Runtime>) -> Result<Pool, CreatePoolError> {
        let mut builder = self.builder().map_err(CreatePoolError::Config)?;
        if let Some(runtime) = runtime {
            builder = builder.runtime(runtime);
        }
        builder.build().map_err(CreatePoolError::Build)
    }

    pub fn builder(&self) -> Result<PoolBuilder, ConfigError> {
        let manager = crate::Manager::new(self.model.clone());
        Ok(Pool::builder(manager).config(self.get_pool_config()))
    }

    #[must_use]
    pub fn get_pool_config(&self) -> PoolConfig {
        self.pool.unwrap_or_default()
    }

    #[must_use]
    pub fn from_model(model: impl Into<ModelKind>) -> Self {
        Self {
            model: model.into(),
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
