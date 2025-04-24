use polars::{error::ErrString, prelude::PolarsError};
use postcard;
use pyo3::PyErr;
use thiserror::Error;

/// Represent possible errors returned by this library.
#[derive(Error, Debug)]
pub enum GramsError {
    /// Represents errors that occur when the input data passing to the library is invalid.
    #[error("Invalid input data: {0}")]
    InvalidInputData(String),
    /// Represents errors that occur when the configuration data passing to the library is invalid.
    #[error("Invalid configuration: {0}")]
    InvalidConfigData(String),

    #[error("Integrity error - asking for an entity that isn't in the database: {0}")]
    DBIntegrityError(String),

    #[error("Integrity error - type error: {0}")]
    DBTypeError(String),

    #[error("DB error - key error: {0}")]
    DBKeyError(String),

    #[error("Generic integrity error: {0}")]
    IntegrityError(String),

    #[error("Logic error: {0}")]
    LogicError(String),

    #[error("Invalid arguments: {0}")]
    InvalidArgument(String),

    #[error(transparent)]
    PostcardError(#[from] postcard::Error),

    /// Represents all other cases of `std::io::Error`.
    #[error(transparent)]
    IOError(#[from] std::io::Error),

    #[error(transparent)]
    KGDataError(#[from] kgdata_core::error::KGDataError),

    /// serde_json error
    #[error(transparent)]
    SerdeJsonErr(#[from] serde_json::Error),

    /// PyO3 error
    #[error(transparent)]
    PyErr(#[from] pyo3::PyErr),

    #[error(transparent)]
    LSAPErr(#[from] lsap::LSAPError),

    #[error(transparent)]
    StrSimError(#[from] yass::StrSimError),

    #[error(transparent)]
    PolarsError(#[from] polars::error::PolarsError),
}

pub fn into_pyerr<E: Into<GramsError>>(err: E) -> PyErr {
    let hderr = err.into();
    if let GramsError::PyErr(e) = hderr {
        e
    } else {
        let anyerror: anyhow::Error = hderr.into();
        anyerror.into()
    }
}

pub fn into_polars_error<E: Into<GramsError>>(err: E) -> PolarsError {
    let hderr = err.into();
    if let GramsError::PolarsError(e) = hderr {
        e
    } else {
        PolarsError::ComputeError(ErrString::from(hderr.to_string()))
    }
}
