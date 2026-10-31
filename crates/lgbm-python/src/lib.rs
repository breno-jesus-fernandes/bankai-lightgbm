//! PyO3 bindings exposed to Python as `bankai_lightgbm._lib`.
//! The public, lightgbm-compatible API lives in `python/bankai_lightgbm`.

use std::collections::HashMap;
use std::sync::Arc;

use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    _lib,
    LightGBMError,
    PyException,
    "Error thrown by LightGBM."
);

fn to_py(err: lgbm_core::Error) -> PyErr {
    LightGBMError::new_err(err.to_string())
}

/// Binned training data. Mirrors upstream's C API `DatasetHandle`.
#[pyclass(module = "bankai_lightgbm._lib", frozen)]
struct NativeDataset {
    ds: Arc<Dataset>,
}

#[pymethods]
impl NativeDataset {
    /// Bin a dense row-major `float64` matrix (`LGBM_DatasetCreateFromMat`). `params` are
    /// the dataset parameters as strings; `categorical` are column indices.
    #[staticmethod]
    #[pyo3(signature = (data, label, params, categorical, weight=None, group=None))]
    fn from_dense(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f64>,
        label: Vec<f64>,
        params: HashMap<String, String>,
        categorical: Vec<i32>,
        weight: Option<Vec<f32>>,
        group: Option<Vec<i32>>,
    ) -> PyResult<Self> {
        let shape = data.shape().to_vec();
        let values = data
            .as_slice()
            .map_err(|_| LightGBMError::new_err("data must be C-contiguous"))?
            .to_vec();
        let ds = py
            .allow_threads(|| -> lgbm_core::Result<Dataset> {
                let cfg = Config::from_params(&params)?;
                let mut ds =
                    Dataset::from_dense(&values, shape[0], shape[1], &label, &cfg, &categorical)?;
                if let Some(w) = weight {
                    ds.set_weights(&w)?;
                }
                if let Some(g) = group {
                    ds.set_group(&g)?;
                }
                Ok(ds)
            })
            .map_err(to_py)?;
        Ok(NativeDataset { ds: Arc::new(ds) })
    }

    fn num_data(&self) -> usize {
        self.ds.num_data()
    }

    fn num_feature(&self) -> usize {
        self.ds.layout.num_total_features
    }
}

/// Native booster handle. Mirrors upstream's C API `BoosterHandle`.
#[pyclass(module = "bankai_lightgbm._lib", frozen)]
struct NativeBooster {
    model: Gbdt,
}

#[pymethods]
impl NativeBooster {
    #[staticmethod]
    fn from_file(py: Python<'_>, path: String) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Gbdt::from_model_file(&path))
            .map_err(to_py)?;
        Ok(NativeBooster { model })
    }

    #[staticmethod]
    fn from_string(py: Python<'_>, model_str: String) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Gbdt::from_model_string(&model_str))
            .map_err(to_py)?;
        Ok(NativeBooster { model })
    }

    /// Train `num_iterations` rounds on `dataset` (`LGBM_BoosterCreate` + `UpdateOneIter`).
    #[staticmethod]
    fn train(
        py: Python<'_>,
        params: HashMap<String, String>,
        dataset: PyRef<'_, NativeDataset>,
    ) -> PyResult<Self> {
        let ds = Arc::clone(&dataset.ds);
        let model = py
            .allow_threads(|| -> lgbm_core::Result<Gbdt> {
                let cfg = Config::from_params(&params)?;
                Gbdt::train_shared(Arc::new(cfg), ds)
            })
            .map_err(to_py)?;
        Ok(NativeBooster { model })
    }

    fn num_class(&self) -> usize {
        self.model.num_class()
    }
    fn num_tree_per_iteration(&self) -> usize {
        self.model.num_tree_per_iteration()
    }
    fn num_trees(&self) -> usize {
        self.model.num_trees()
    }
    fn current_iteration(&self) -> usize {
        self.model.current_iteration()
    }
    fn num_feature(&self) -> usize {
        self.model.num_features()
    }
    fn feature_names(&self) -> Vec<String> {
        self.model.feature_names().to_vec()
    }
    fn pandas_categorical(&self) -> Option<String> {
        self.model.pandas_categorical().map(str::to_string)
    }
    fn objective(&self) -> Option<String> {
        self.model.objective().map(|o| o.raw.clone())
    }

    fn save_model_to_string(
        &self,
        py: Python<'_>,
        start_iteration: i32,
        num_iteration: i32,
        importance_type: i32,
    ) -> PyResult<String> {
        py.allow_threads(|| {
            self.model
                .save_model_to_string(start_iteration, num_iteration, importance_type)
        })
        .map_err(to_py)
    }

    fn dump_model(
        &self,
        py: Python<'_>,
        start_iteration: i32,
        num_iteration: i32,
        importance_type: i32,
    ) -> PyResult<String> {
        py.allow_threads(|| {
            self.model
                .dump_model(start_iteration, num_iteration, importance_type)
        })
        .map_err(to_py)
    }

    fn feature_importance(&self, num_iteration: i32, importance_type: i32) -> PyResult<Vec<f64>> {
        self.model
            .feature_importance(num_iteration, importance_type)
            .map_err(to_py)
    }

    fn upper_bound(&self) -> f64 {
        self.model.upper_bound_value()
    }

    fn lower_bound(&self) -> f64 {
        self.model.lower_bound_value()
    }

    fn leaf_value(&self, tree_idx: usize, leaf_idx: usize) -> PyResult<f64> {
        self.model
            .leaf_value(tree_idx, leaf_idx)
            .ok_or_else(|| LightGBMError::new_err("tree or leaf index out of range"))
    }

    /// Predict a C-contiguous float64 matrix. `kind`: 0 normal, 1 raw, 2 leaf, 3 contrib
    /// (upstream `C_API_PREDICT_*`). Returns `nrow x num_pred_one_row`.
    #[pyo3(signature = (data, kind, start_iteration, num_iteration, pred_early_stop=false,
        pred_early_stop_freq=10, pred_early_stop_margin=10.0, num_threads=0, disable_shape_check=false))]
    #[allow(clippy::too_many_arguments)]
    fn predict_dense<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        kind: i32,
        start_iteration: i32,
        num_iteration: i32,
        pred_early_stop: bool,
        pred_early_stop_freq: i32,
        pred_early_stop_margin: f64,
        num_threads: i32,
        disable_shape_check: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = data.shape();
        let (nrow, ncol) = (shape[0], shape[1]);
        if !disable_shape_check && ncol != self.model.num_features() {
            return Err(LightGBMError::new_err(format!(
                "The number of features in data ({ncol}) is not the same as it was in training data ({}).\n\
                 You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.",
                self.model.num_features()
            )));
        }
        let kind = match kind {
            0 => PredictKind::Normal,
            1 => PredictKind::RawScore,
            2 => PredictKind::LeafIndex,
            3 => PredictKind::Contrib,
            other => {
                return Err(LightGBMError::new_err(format!(
                    "unknown predict type {other}"
                )));
            }
        };
        let cfg = PredictConfig {
            kind,
            start_iteration,
            num_iteration,
            pred_early_stop,
            pred_early_stop_freq,
            pred_early_stop_margin,
            num_threads: usize::try_from(num_threads).unwrap_or(0),
        };
        let slice = data
            .as_slice()
            .map_err(|e| LightGBMError::new_err(e.to_string()))?;
        let model = &self.model;
        let (out, width) = py
            .allow_threads(|| {
                let p = Predictor::new(model, &cfg)?;
                let w = p.num_pred_one_row();
                p.predict_dense(slice, nrow, ncol).map(|o| (o, w))
            })
            .map_err(to_py)?;
        let arr = numpy::PyArray1::from_vec(py, out);
        arr.reshape([nrow, width])
    }
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("UPSTREAM_VERSION", lgbm_core::UPSTREAM_VERSION)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("LightGBMError", m.py().get_type::<LightGBMError>())?;
    m.add_class::<NativeBooster>()?;
    m.add_class::<NativeDataset>()?;
    Ok(())
}
