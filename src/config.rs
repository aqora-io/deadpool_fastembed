use crate::{CreatePoolError, Pool, PoolBuilder, PoolConfig, Runtime};
use fastembed::{
    InitOptionsUserDefined, RerankInitOptionsUserDefined, SparseTextEmbedding, TextEmbedding,
    TextRerank, UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};

#[cfg(feature = "image-models")]
use fastembed::{
    ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, ImageInitOptionsUserDefined,
    UserDefinedImageEmbeddingModel,
};

#[cfg(feature = "hf-hub")]
use fastembed::{
    EmbeddingModel, RerankInitOptions, RerankerModel, SparseInitOptions, SparseModel,
    TextInitOptions,
};

#[derive(Debug, Clone)]
#[cfg(feature = "hf-hub")]
pub enum ModelKind {
    Text(TextInitOptions),
    #[cfg(feature = "image-models")]
    Image(ImageInitOptions),
    Sparse(SparseInitOptions),
    Rerank(RerankInitOptions),
}

#[cfg(feature = "hf-hub")]
impl Default for ModelKind {
    fn default() -> Self {
        Self::text(EmbeddingModel::default())
    }
}

#[cfg(feature = "hf-hub")]
impl ModelKind {
    pub fn text(model: EmbeddingModel) -> Self {
        Self::Text(TextInitOptions::new(model))
    }

    #[cfg(feature = "image-models")]
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

#[derive(Debug, Clone)]
pub enum UserDefinedModelKind {
    Text {
        options: InitOptionsUserDefined,
        model: UserDefinedEmbeddingModel,
    },
    #[cfg(feature = "image-models")]
    Image {
        options: ImageInitOptionsUserDefined,
        model: UserDefinedImageEmbeddingModel,
    },
    Rerank {
        options: RerankInitOptionsUserDefined,
        model: UserDefinedRerankingModel,
    },
}

impl UserDefinedModelKind {
    pub fn text(model: UserDefinedEmbeddingModel) -> Self {
        Self::Text {
            options: InitOptionsUserDefined::new(),
            model,
        }
    }
    #[cfg(feature = "image-models")]
    pub fn image(model: UserDefinedImageEmbeddingModel) -> Self {
        Self::Image {
            options: ImageInitOptionsUserDefined::new(),
            model,
        }
    }
    pub fn rerank(model: UserDefinedRerankingModel) -> Self {
        Self::Rerank {
            options: RerankInitOptionsUserDefined::default(),
            model,
        }
    }
}

pub enum EmbeddingKind {
    Text(TextEmbedding),
    #[cfg(feature = "image-models")]
    Image(ImageEmbedding),
    Sparse(SparseTextEmbedding),
    ReRanking(TextRerank),
}

impl EmbeddingKind {
    #[cfg(feature = "hf-hub")]
    pub fn try_new(model: &ModelKind) -> Result<Self, fastembed::Error> {
        Ok(match model.to_owned() {
            ModelKind::Text(opts) => Self::Text(TextEmbedding::try_new(opts)?),
            #[cfg(feature = "image-models")]
            ModelKind::Image(opts) => Self::Image(ImageEmbedding::try_new(opts)?),
            ModelKind::Sparse(opts) => Self::Sparse(SparseTextEmbedding::try_new(opts)?),
            ModelKind::Rerank(opts) => Self::ReRanking(TextRerank::try_new(opts)?),
        })
    }

    pub fn try_new_from_user_defined(
        model: &UserDefinedModelKind,
    ) -> Result<Self, fastembed::Error> {
        Ok(match model.to_owned() {
            UserDefinedModelKind::Text { options, model } => Self::Text(
                TextEmbedding::try_new_from_user_defined(model.to_owned(), options.to_owned())?,
            ),
            #[cfg(feature = "image-models")]
            UserDefinedModelKind::Image { options, model } => Self::Image(
                ImageEmbedding::try_new_from_user_defined(model.to_owned(), options.to_owned())?,
            ),
            UserDefinedModelKind::Rerank { options, model } => Self::ReRanking(
                TextRerank::try_new_from_user_defined(model.to_owned(), options.to_owned())?,
            ),
        })
    }
}

/// Configuration object.
#[derive(Clone, Debug)]
pub struct Config {
    #[cfg(feature = "hf-hub")]
    pub model: ModelKind,
    #[cfg(not(feature = "hf-hub"))]
    pub model: UserDefinedModelKind,

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
    #[cfg(feature = "hf-hub")]
    pub fn from_model(model: impl Into<ModelKind>) -> Self {
        Self {
            model: model.into(),
            pool: None,
        }
    }

    #[must_use]
    #[cfg(not(feature = "hf-hub"))]
    pub fn from_model(model: impl Into<UserDefinedModelKind>) -> Self {
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
