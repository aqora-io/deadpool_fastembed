use deadpool::managed;
use fastembed::Error as FastembedError;

mod config;

#[cfg(not(feature = "hf-hub"))]
use crate::config::UserDefinedModelKind;
use config::ConfigError;
#[cfg(feature = "hf-hub")]
pub use config::ModelKind;
pub use config::{Config, EmbeddingKind};
pub use deadpool::managed::reexports::*;

deadpool::managed_reexports!("fastembed", Manager, Embedding, FastembedError, ConfigError);

/// Type alias for using [`deadpool::managed::RecycleResult`] with [`fastembed`].
type RecycleResult = managed::RecycleResult<FastembedError>;

pub struct Embedding(Object);

impl From<Object> for Embedding {
    fn from(value: Object) -> Self {
        Self(value)
    }
}

impl std::ops::Deref for Embedding {
    type Target = EmbeddingKind;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Embedding {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Embedding {
    pub fn as_text_mut(&mut self) -> Option<&mut fastembed::TextEmbedding> {
        if let EmbeddingKind::Text(te) = &mut **self {
            Some(te)
        } else {
            None
        }
    }

    #[cfg(feature = "image-models")]
    pub fn as_image_mut(&mut self) -> Option<&mut fastembed::ImageEmbedding> {
        if let EmbeddingKind::Image(te) = &mut **self {
            Some(te)
        } else {
            None
        }
    }
    pub fn as_sparse_mut(&mut self) -> Option<&mut fastembed::SparseTextEmbedding> {
        if let EmbeddingKind::Sparse(te) = &mut **self {
            Some(te)
        } else {
            None
        }
    }
    pub fn as_reranker_mut(&mut self) -> Option<&mut fastembed::TextRerank> {
        if let EmbeddingKind::ReRanking(te) = &mut **self {
            Some(te)
        } else {
            None
        }
    }
}

/// [`Manager`] for creating and recycling fastembed.
///
/// [`Manager`]: managed::Manager
#[derive(Debug)]
pub struct Manager {
    #[cfg(feature = "hf-hub")]
    model: ModelKind,
    #[cfg(not(feature = "hf-hub"))]
    model: UserDefinedModelKind,
}

impl Manager {
    #[cfg(feature = "hf-hub")]
    pub fn new(model: ModelKind) -> Self {
        Self { model }
    }

    #[cfg(not(feature = "hf-hub"))]
    pub fn new(model: UserDefinedModelKind) -> Self {
        Self { model }
    }
}

impl managed::Manager for Manager {
    type Type = EmbeddingKind;
    type Error = fastembed::Error;

    #[cfg(feature = "hf-hub")]
    async fn create(&self) -> Result<Self::Type, Self::Error> {
        EmbeddingKind::try_new(&self.model)
    }

    #[cfg(not(feature = "hf-hub"))]
    async fn create(&self) -> Result<Self::Type, Self::Error> {
        EmbeddingKind::try_new_from_user_defined(&self.model)
    }

    async fn recycle(&self, _: &mut Self::Type, _: &managed::Metrics) -> RecycleResult {
        Ok(())
    }
}
