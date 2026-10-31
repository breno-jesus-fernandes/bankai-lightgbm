//! Objective functions. P1 covers the prediction side: turning a model file's
//! `objective=` line into an output transform. upstream: src/objective/

pub mod binary;
pub mod multiclass;
pub mod regression;
pub mod train;
pub mod xentropy;

use crate::error::{Error, Result};
use crate::utils::common::{parse_scalar, softmax};

/// upstream: `ParseObjectiveAlias` (include/LightGBM/config.h)
pub fn parse_objective_alias(name: &str) -> &str {
    match name {
        "regression"
        | "regression_l2"
        | "mean_squared_error"
        | "mse"
        | "l2"
        | "l2_root"
        | "root_mean_squared_error"
        | "rmse" => "regression",
        "regression_l1" | "mean_absolute_error" | "l1" | "mae" => "regression_l1",
        "multiclass" | "softmax" => "multiclass",
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => "multiclassova",
        "xentropy" | "cross_entropy" => "cross_entropy",
        "xentlambda" | "cross_entropy_lambda" => "cross_entropy_lambda",
        "mean_absolute_percentage_error" | "mape" => "mape",
        "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart" | "xendcg_mart" => "rank_xendcg",
        "none" | "null" | "custom" | "na" => "custom",
        other => other,
    }
}

/// upstream: `Common::Sign` — `(x > 0) - (x < 0)` as an int (0 for zero and NaN).
#[inline]
fn sign(x: f64) -> f64 {
    f64::from(i32::from(x > 0.0) - i32::from(x < 0.0))
}

/// Raw-score -> prediction transform of a loaded objective (`ConvertOutput`).
#[derive(Debug, Clone, PartialEq)]
pub enum OutputTransform {
    Identity,
    /// `sign(x) * x^2` (regression objectives trained with `reg_sqrt`).
    SignedSquare,
    Exp,
    Sigmoid(f64),
    Softmax,
    /// Per-class sigmoid (multiclassova).
    SigmoidEach(f64),
    /// `log1p(exp(x))` (cross_entropy_lambda).
    Log1pExp,
}

/// Prediction-side view of an objective, parsed from the model's `objective=` line.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedObjective {
    pub name: String,
    pub transform: OutputTransform,
    pub need_accurate_prediction: bool,
    /// The full serialized string as loaded.
    pub raw: String,
    pub sqrt: bool,
    pub num_class: i32,
}

impl LoadedObjective {
    /// upstream: `ObjectiveFunction::CreateObjectiveFunction(const std::string&)`.
    /// Returns `None` for `custom`.
    pub fn from_model_str(s: &str) -> Result<Option<Self>> {
        let mut toks = s.split(' ').filter(|t| !t.is_empty());
        let ty = parse_objective_alias(toks.next().unwrap_or("")).to_string();
        let rest: Vec<&str> = toks.collect();
        let sqrt = rest.contains(&"sqrt");
        let kv = |key: &str| {
            rest.iter()
                .filter_map(|t| t.split_once(':'))
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v)
        };
        let sigmoid = || -> Result<f64> {
            let v: f64 = parse_scalar(kv("sigmoid").unwrap_or("-1"))?;
            if v <= 0.0 {
                return Err(Error::Parse(format!(
                    "Sigmoid parameter {v} should be greater than zero"
                )));
            }
            Ok(v)
        };
        let num_class: i32 = match kv("num_class") {
            Some(v) => parse_scalar(v)?,
            None => 0,
        };
        let transform = match ty.as_str() {
            "regression" | "regression_l1" | "quantile" | "fair" | "mape" => {
                if sqrt {
                    OutputTransform::SignedSquare
                } else {
                    OutputTransform::Identity
                }
            }
            // Huber disables sqrt; poisson/gamma/tweedie disable it at Init and use exp.
            "huber" => OutputTransform::Identity,
            "poisson" | "gamma" | "tweedie" => OutputTransform::Exp,
            "binary" => OutputTransform::Sigmoid(sigmoid()?),
            "multiclass" => {
                kv("num_class").ok_or_else(|| {
                    Error::Parse("Objective should contain num_class field".into())
                })?;
                OutputTransform::Softmax
            }
            "multiclassova" => {
                kv("num_class").ok_or_else(|| {
                    Error::Parse("Objective should contain num_class field".into())
                })?;
                OutputTransform::SigmoidEach(sigmoid()?)
            }
            "cross_entropy" => OutputTransform::Sigmoid(1.0),
            "cross_entropy_lambda" => OutputTransform::Log1pExp,
            "lambdarank" | "rank_xendcg" => OutputTransform::Identity,
            "custom" => return Ok(None),
            other => {
                return Err(Error::Parse(format!(
                    "Unknown objective type name: {other}"
                )));
            }
        };
        let need_accurate_prediction = !matches!(
            ty.as_str(),
            "binary" | "multiclass" | "multiclassova" | "lambdarank" | "rank_xendcg"
        );
        // Huber disables `sqrt` in its constructor, so it never round-trips.
        let sqrt = sqrt && ty != "huber";
        Ok(Some(LoadedObjective {
            name: ty,
            transform,
            need_accurate_prediction,
            raw: s.to_string(),
            sqrt,
            num_class,
        }))
    }

    /// upstream: `ObjectiveFunction::ToString` for each objective.
    pub fn to_model_string(&self) -> String {
        use crate::utils::format::format_g;
        match (&self.transform, self.name.as_str()) {
            (OutputTransform::Sigmoid(s), "binary") => {
                format!("binary sigmoid:{}", format_g(*s, 6))
            }
            (_, "multiclass") => format!("multiclass num_class:{}", self.num_class),
            (OutputTransform::SigmoidEach(s), "multiclassova") => format!(
                "multiclassova num_class:{} sigmoid:{}",
                self.num_class,
                format_g(*s, 6)
            ),
            (_, name) if self.sqrt => format!("{name} sqrt"),
            (_, name) => name.to_string(),
        }
    }

    /// upstream: `ObjectiveFunction::ConvertOutput`, applied in place.
    #[inline]
    pub fn convert_output(&self, io: &mut [f64]) {
        match self.transform {
            OutputTransform::Identity => {}
            OutputTransform::SignedSquare => io[0] = sign(io[0]) * io[0] * io[0],
            OutputTransform::Exp => io[0] = io[0].exp(),
            // `1.0f / (1.0f + std::exp(-sigmoid_ * x))`: float literals promote to double.
            OutputTransform::Sigmoid(s) => io[0] = 1.0 / (1.0 + (-s * io[0]).exp()),
            OutputTransform::Softmax => softmax(io),
            OutputTransform::SigmoidEach(s) => {
                for v in io.iter_mut() {
                    *v = 1.0 / (1.0 + (-s * *v).exp());
                }
            }
            OutputTransform::Log1pExp => io[0] = io[0].exp().ln_1p(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn parses_binary_with_sigmoid() {
        let o = LoadedObjective::from_model_str("binary sigmoid:2")
            .unwrap()
            .unwrap();
        assert_eq!(o.transform, OutputTransform::Sigmoid(2.0));
        let mut v = [0.0];
        o.convert_output(&mut v);
        assert_eq!(v[0], 0.5);
    }

    #[test]
    fn binary_without_sigmoid_is_an_error() {
        assert!(LoadedObjective::from_model_str("binary").is_err());
    }

    #[test]
    fn aliases_and_sqrt() {
        let o = LoadedObjective::from_model_str("mse sqrt")
            .unwrap()
            .unwrap();
        assert_eq!(o.name, "regression");
        let mut v = [-3.0];
        o.convert_output(&mut v);
        assert_eq!(v[0], -9.0);
    }

    #[test]
    fn custom_has_no_transform() {
        assert!(LoadedObjective::from_model_str("custom").unwrap().is_none());
    }

    #[test]
    fn to_model_string_matches_upstream_tostring() {
        for s in [
            "binary sigmoid:1",
            "binary sigmoid:0.5",
            "multiclass num_class:3",
            "multiclassova num_class:3 sigmoid:1",
            "regression sqrt",
            "regression",
            "lambdarank",
            "cross_entropy",
            "tweedie",
        ] {
            let o = LoadedObjective::from_model_str(s).unwrap().unwrap();
            assert_eq!(o.to_model_string(), s);
        }
        let o = LoadedObjective::from_model_str("huber sqrt")
            .unwrap()
            .unwrap();
        assert_eq!(o.to_model_string(), "huber");
    }

    #[test]
    fn unknown_is_an_error() {
        assert!(LoadedObjective::from_model_str("nope").is_err());
    }
}
