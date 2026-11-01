//! Error type shared by all bankai-lightgbm crates.

/// Errors raised by bankai-lightgbm. Mirrors upstream's `LightGBMError`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid parameter value or combination.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// Feature present upstream but not yet ported; never silently ignored.
    #[error("not supported yet: {0}")]
    Unsupported(String),
    /// Malformed model or data input.
    #[error("parse error: {0}")]
    Parse(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_message_names_the_feature() {
        let err = Error::Unsupported("device_type=cuda".into());
        assert_eq!(err.to_string(), "not supported yet: device_type=cuda");
    }
}
