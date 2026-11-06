//! PyO3 bindings exposed to Python as `bankai_lightgbm._lib`.
//! The public, lightgbm-compatible API lives in `python/bankai_lightgbm`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lgbm_core::boosting::{Gbdt, GbdtTrainer};
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::io::sparse::{CsrMatrix, csc_to_csr};
use lgbm_core::metric::create_metric;
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
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
    data: Arc<RawRows>,
    nrow: usize,
    ncol: usize,
}

/// The raw feature rows a dataset was built from, kept to score validation sets.
enum RawRows {
    /// Row-major `nrow x ncol`.
    Dense(Vec<f64>),
    /// CSR arrays (`indptr`, `indices`, `values`), in the caller's storage order.
    Csr(Vec<i64>, Vec<i32>, Vec<f64>),
    /// Binned only (a binary dataset file keeps no raw values): training works, but scoring
    /// rows with this dataset is not supported yet.
    Unavailable,
}

impl RawRows {
    /// The rows `indices`, in that order.
    fn select(&self, indices: &[usize], ncol: usize) -> RawRows {
        match self {
            RawRows::Dense(data) => {
                let mut out = Vec::with_capacity(indices.len() * ncol);
                for &row in indices {
                    out.extend_from_slice(&data[row * ncol..(row + 1) * ncol]);
                }
                RawRows::Dense(out)
            }
            RawRows::Csr(indptr, cols, vals) => {
                let mut out_ptr = Vec::with_capacity(indices.len() + 1);
                out_ptr.push(0i64);
                let (mut out_cols, mut out_vals) = (Vec::new(), Vec::new());
                for &row in indices {
                    let (lo, hi) = (indptr[row] as usize, indptr[row + 1] as usize);
                    out_cols.extend_from_slice(&cols[lo..hi]);
                    out_vals.extend_from_slice(&vals[lo..hi]);
                    out_ptr.push(out_cols.len() as i64);
                }
                RawRows::Csr(out_ptr, out_cols, out_vals)
            }
            RawRows::Unavailable => RawRows::Unavailable,
        }
    }

    /// Raw scores for every row, row-major (`nrow x num_class`).
    fn predict(
        &self,
        predictor: &Predictor<'_>,
        nrow: usize,
        ncol: usize,
    ) -> lgbm_core::Result<Vec<f64>> {
        match self {
            RawRows::Dense(data) => predictor.predict_dense(data, nrow, ncol),
            RawRows::Csr(indptr, cols, vals) => predictor.predict_csr(indptr, cols, vals),
            RawRows::Unavailable => Err(lgbm_core::Error::Unsupported(
                "scoring a validation set built from a binary dataset file is not supported yet"
                    .into(),
            )),
        }
    }
}

#[pymethods]
impl NativeDataset {
    /// Load a binary dataset file written by `save_binary` (upstream `DatasetLoader::LoadFromBinFile`).
    #[staticmethod]
    fn from_binary_file(py: Python<'_>, path: String) -> PyResult<Self> {
        let ds = py
            .allow_threads(|| -> lgbm_core::Result<Dataset> {
                let bin = lgbm_core::io::binary::BinaryDataset::read_file(&path)?;
                Dataset::from_binary(&bin)
            })
            .map_err(to_py)?;
        let (nrow, ncol) = (ds.num_data(), ds.layout.num_total_features);
        Ok(NativeDataset {
            ds: Arc::new(ds),
            data: Arc::new(RawRows::Unavailable),
            nrow,
            ncol,
        })
    }

    /// Bin a dense row-major `float64` matrix (`LGBM_DatasetCreateFromMat`). `params` are
    /// the dataset parameters as strings; `categorical` are column indices.
    #[staticmethod]
    #[pyo3(signature = (data, label, params, categorical, weight=None, group=None, reference=None, init_score=None, feature_names=None, two_round=false))]
    #[allow(clippy::too_many_arguments)] // Python Dataset constructor arguments.
    fn from_dense(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f64>,
        label: Vec<f64>,
        params: HashMap<String, String>,
        categorical: Vec<i32>,
        weight: Option<Vec<f32>>,
        group: Option<Vec<i32>>,
        reference: Option<PyRef<'_, NativeDataset>>,
        init_score: Option<Vec<f64>>,
        feature_names: Option<Vec<String>>,
        two_round: bool,
    ) -> PyResult<Self> {
        let shape = data.shape().to_vec();
        let values = data
            .as_slice()
            .map_err(|_| LightGBMError::new_err("data must be C-contiguous"))?
            .to_vec();
        let (nrow, ncol) = (shape[0], shape[1]);
        let reference = reference.map(|r| Arc::clone(&r.ds));
        let ds = py
            .allow_threads(|| -> lgbm_core::Result<Dataset> {
                let cfg = Config::from_params(&params)?;
                let ds = if let Some(reference) = reference {
                    Dataset::from_dense_reference(&values, nrow, ncol, &label, &reference)?
                } else if two_round {
                    Dataset::from_dense_two_round(&values, nrow, ncol, &label, &cfg, &categorical)?
                } else {
                    Dataset::from_dense(&values, nrow, ncol, &label, &cfg, &categorical)?
                };
                let mut ds = with_metadata(ds, weight, group, init_score)?;
                apply_feature_names(&mut ds, feature_names)?;
                Ok(ds)
            })
            .map_err(to_py)?;
        Ok(NativeDataset {
            ds: Arc::new(ds),
            data: Arc::new(RawRows::Dense(values)),
            nrow,
            ncol,
        })
    }

    /// Bin a scipy CSR (`is_csc=false`) or CSC (`is_csc=true`) matrix of shape
    /// `nrow x ncol` (`LGBM_DatasetCreateFromCSR`/`FromCSC`). CSC input is gathered into CSR
    /// rows in column order first. Other arguments are as for `from_dense`.
    #[staticmethod]
    #[pyo3(signature = (indptr, indices, data, nrow, ncol, is_csc, label, params, categorical,
        weight=None, group=None, reference=None, init_score=None, feature_names=None))]
    #[allow(clippy::too_many_arguments)] // Python Dataset constructor arguments.
    fn from_sparse(
        py: Python<'_>,
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f64>,
        nrow: usize,
        ncol: usize,
        is_csc: bool,
        label: Vec<f64>,
        params: HashMap<String, String>,
        categorical: Vec<i32>,
        weight: Option<Vec<f32>>,
        group: Option<Vec<i32>>,
        reference: Option<PyRef<'_, NativeDataset>>,
        init_score: Option<Vec<f64>>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let contiguous = |e: numpy::NotContiguousError| LightGBMError::new_err(e.to_string());
        let (indptr, indices, values) = (
            indptr.as_slice().map_err(contiguous)?.to_vec(),
            indices.as_slice().map_err(contiguous)?.to_vec(),
            data.as_slice().map_err(contiguous)?.to_vec(),
        );
        let reference = reference.map(|r| Arc::clone(&r.ds));
        let (ds, raw) = py
            .allow_threads(|| -> lgbm_core::Result<(Dataset, RawRows)> {
                let (indptr, indices, values) = if is_csc {
                    csc_to_csr(&indptr, &indices, &values, nrow)?
                } else {
                    (indptr, indices, values)
                };
                let cfg = Config::from_params(&params)?;
                let m = CsrMatrix::new(&indptr, &indices, &values, ncol)?;
                let ds = if let Some(reference) = reference {
                    Dataset::from_csr_reference(&m, &label, &reference)?
                } else {
                    Dataset::from_csr(&m, &label, &cfg, &categorical)?
                };
                let mut ds = with_metadata(ds, weight, group, init_score)?;
                apply_feature_names(&mut ds, feature_names)?;
                Ok((ds, RawRows::Csr(indptr, indices, values)))
            })
            .map_err(to_py)?;
        Ok(NativeDataset {
            ds: Arc::new(ds),
            data: Arc::new(raw),
            nrow,
            ncol,
        })
    }

    fn num_data(&self) -> usize {
        self.ds.num_data()
    }

    /// Write the bin dump to `path` (upstream `Dataset::DumpTextFile`).
    fn dump_text(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, lgbm_core::io::dump::dump_text(&self.ds))
            .map_err(|err| pyo3::exceptions::PyOSError::new_err(err.to_string()))
    }

    /// A copy with any given metadata field replaced (`None` keeps it; an empty list clears).
    fn with_fields(
        &self,
        label: Option<Vec<f64>>,
        weight: Option<Vec<f32>>,
        group: Option<Vec<i32>>,
    ) -> PyResult<Self> {
        let mut ds = (*self.ds).clone();
        if let Some(label) = label {
            ds.set_label(&label).map_err(to_py)?;
        }
        if let Some(weight) = weight {
            ds.set_weights(&weight).map_err(to_py)?;
        }
        if let Some(group) = group {
            ds.set_group(&group).map_err(to_py)?;
        }
        Ok(NativeDataset {
            ds: Arc::new(ds),
            data: Arc::clone(&self.data),
            nrow: self.nrow,
            ncol: self.ncol,
        })
    }

    /// A copy of this dataset with the init score replaced (class-major; empty clears it).
    /// The binned data is copied; the raw matrix is shared.
    fn with_init_score(&self, init_score: Vec<f64>) -> PyResult<Self> {
        let mut ds = (*self.ds).clone();
        ds.set_init_score(&init_score).map_err(to_py)?;
        Ok(NativeDataset {
            ds: Arc::new(ds),
            data: Arc::clone(&self.data),
            nrow: self.nrow,
            ncol: self.ncol,
        })
    }

    fn num_feature(&self) -> usize {
        self.ds.layout.num_total_features
    }

    /// Bins of one feature (upstream `LGBM_DatasetGetFeatureNumBin`); out-of-range indices raise.
    fn feature_num_bin(&self, feature: i64) -> PyResult<i32> {
        let total = self.ds.layout.num_total_features;
        if feature < 0 || feature as usize >= total {
            return Err(LightGBMError::new_err(format!(
                "Tried to retrieve number of bins for feature index {feature}, \
                 but the valid feature indices are [0, {}].",
                total as i64 - 1
            )));
        }
        Ok(self.ds.feature_num_bin(feature as usize))
    }

    /// Labels as stored natively (float32, NaN mapped to 0 and clamped like upstream's
    /// `Metadata::SetLabel`), for `Dataset.get_label` after construction.
    fn label<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        numpy::PyArray1::from_slice(py, &self.ds.metadata.label)
    }

    /// Weights as stored natively (sanitized like upstream), `None` when unweighted.
    fn weight<'py>(&self, py: Python<'py>) -> Option<Bound<'py, numpy::PyArray1<f32>>> {
        let w = &self.ds.metadata.weights;
        (!w.is_empty()).then(|| numpy::PyArray1::from_slice(py, w))
    }

    /// Write the binned dataset in upstream's binary format (`LGBM_DatasetSaveBinary`).
    fn save_binary(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let ds = Arc::clone(&self.ds);
        py.allow_threads(|| ds.save_binary(&path)).map_err(to_py)
    }

    fn subset(&self, indices: Vec<usize>) -> PyResult<Self> {
        let ds = self.ds.subset(&indices).map_err(to_py)?;
        Ok(NativeDataset {
            ds: Arc::new(ds),
            data: Arc::new(self.data.select(&indices, self.ncol)),
            nrow: indices.len(),
            ncol: self.ncol,
        })
    }
}

/// Native booster handle. Mirrors upstream's C API `BoosterHandle`.
#[pyclass(module = "bankai_lightgbm._lib")]
struct NativeBooster {
    model: Gbdt,
    trainer: Option<GbdtTrainer>,
}

#[pymethods]
impl NativeBooster {
    #[staticmethod]
    fn from_file(py: Python<'_>, path: String) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Gbdt::from_model_file(&path))
            .map_err(to_py)?;
        Ok(NativeBooster {
            model,
            trainer: None,
        })
    }

    #[staticmethod]
    fn from_string(py: Python<'_>, model_str: String) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Gbdt::from_model_string(&model_str))
            .map_err(to_py)?;
        Ok(NativeBooster {
            model,
            trainer: None,
        })
    }

    /// Train `num_iterations` rounds on `dataset` (`LGBM_BoosterCreate` + `UpdateOneIter`).
    #[staticmethod]
    #[pyo3(signature = (params, dataset, init_model=None, warn_config=true))]
    fn train(
        py: Python<'_>,
        params: HashMap<String, String>,
        dataset: PyRef<'_, NativeDataset>,
        init_model: Option<PyRef<'_, NativeBooster>>,
        warn_config: bool,
    ) -> PyResult<Self> {
        let ds = Arc::clone(&dataset.ds);
        let init = init_model.map(|m| m.model.clone());
        let (trainer, model) = py
            .allow_threads(|| -> lgbm_core::Result<(GbdtTrainer, Gbdt)> {
                let cfg = {
                    let _mute = MuteGuard::new(!warn_config);
                    Config::from_params(&params)?
                };
                let cfg = Arc::new(cfg);
                let mut trainer = match &init {
                    Some(init) => GbdtTrainer::with_init_model(Arc::clone(&cfg), ds, init)?,
                    None => GbdtTrainer::new(Arc::clone(&cfg), ds)?,
                };
                for _ in 0..cfg.num_iterations {
                    if trainer.update_one_iter() {
                        break;
                    }
                }
                let model = trainer.snapshot();
                Ok((trainer, model))
            })
            .map_err(to_py)?;
        Ok(NativeBooster {
            model,
            trainer: Some(trainer),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (params, dataset, init_model=None))]
    fn create_training(
        py: Python<'_>,
        params: HashMap<String, String>,
        dataset: PyRef<'_, NativeDataset>,
        init_model: Option<PyRef<'_, NativeBooster>>,
    ) -> PyResult<Self> {
        let ds = Arc::clone(&dataset.ds);
        let init = init_model.map(|m| m.model.clone());
        let trainer = py
            .allow_threads(|| -> lgbm_core::Result<GbdtTrainer> {
                let cfg = Arc::new(Config::from_params(&params)?);
                match &init {
                    Some(init) => GbdtTrainer::with_init_model(cfg, ds, init),
                    None => GbdtTrainer::new(cfg, ds),
                }
            })
            .map_err(to_py)?;
        let model = trainer.snapshot();
        Ok(NativeBooster {
            model,
            trainer: Some(trainer),
        })
    }

    /// Refit the trees of `source` on `dataset` (upstream `LGBM_BoosterMerge` followed by
    /// `LGBM_BoosterRefit`). `leaf_pred` holds the leaf of every row in every tree, row-major
    /// with `ncol` columns (one per tree).
    #[staticmethod]
    #[pyo3(signature = (params, dataset, source, leaf_pred, ncol))]
    fn refit(
        py: Python<'_>,
        params: HashMap<String, String>,
        dataset: PyRef<'_, NativeDataset>,
        source: PyRef<'_, NativeBooster>,
        leaf_pred: Vec<i32>,
        ncol: usize,
    ) -> PyResult<Self> {
        let ds = Arc::clone(&dataset.ds);
        let init = source.model.clone();
        let trainer = py
            .allow_threads(|| -> lgbm_core::Result<GbdtTrainer> {
                let cfg = Arc::new(Config::from_params(&params)?);
                GbdtTrainer::refit(cfg, ds, &init, &leaf_pred, ncol)
            })
            .map_err(to_py)?;
        let model = trainer.snapshot();
        Ok(NativeBooster {
            model,
            trainer: Some(trainer),
        })
    }

    /// Shuffle iterations in `[start, end)` in the model and the trainer (upstream
    /// `LGBM_BoosterShuffleModels`).
    fn shuffle_models(&mut self, start: i32, end: i32) {
        self.model.shuffle_models(start, end);
        if let Some(trainer) = self.trainer.as_mut() {
            trainer.shuffle_models(start, end);
        }
    }

    /// Edit one leaf output in the model and in the trainer's copy (upstream
    /// `LGBM_BoosterSetLeafValue`), so continued training sees the edit.
    fn set_leaf_output(&mut self, tree_idx: usize, leaf_idx: usize, value: f64) -> PyResult<()> {
        self.model
            .set_leaf_value(tree_idx, leaf_idx, value)
            .map_err(to_py)?;
        if let Some(trainer) = self.trainer.as_mut() {
            trainer
                .set_leaf_value(tree_idx, leaf_idx, value)
                .map_err(to_py)?;
        }
        Ok(())
    }

    /// Drop the last iteration (upstream `LGBM_BoosterRollbackOneIter`).
    fn rollback_one_iter(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(trainer) = self.trainer.as_mut() else {
            return Err(LightGBMError::new_err(
                "Cannot roll back a Booster loaded from a model; construct it with a training Dataset.",
            ));
        };
        let model = py.allow_threads(|| -> lgbm_core::Result<Gbdt> {
            trainer.rollback_one_iter()?;
            Ok(trainer.snapshot())
        });
        self.model = model.map_err(to_py)?;
        Ok(())
    }

    fn update(&mut self, py: Python<'_>) -> PyResult<bool> {
        let Some(trainer) = self.trainer.as_mut() else {
            return Err(LightGBMError::new_err(
                "Cannot continue training a Booster loaded from a model; construct it with a training Dataset.",
            ));
        };
        if self.model.objective().is_none() {
            return Err(LightGBMError::new_err(
                "Cannot update due to null objective function.",
            ));
        }
        let (finished, model) = py.allow_threads(|| {
            let finished = trainer.update_one_iter();
            (finished, trainer.snapshot())
        });
        self.model = model;
        Ok(finished)
    }

    fn reset_custom_objective(&mut self) -> PyResult<()> {
        let trainer = self
            .trainer
            .as_mut()
            .ok_or_else(|| LightGBMError::new_err("Booster has no training Dataset"))?;
        trainer.reset_custom_objective().map_err(to_py)?;
        self.model = trainer.snapshot();
        Ok(())
    }

    /// The raw training score handed to a custom objective (class-major, init score
    /// included; DART drops this iteration's trees first), as `LGBM_BoosterGetPredict(0)`.
    fn training_score<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let trainer = self
            .trainer
            .as_mut()
            .ok_or_else(|| LightGBMError::new_err("Booster has no training Dataset"))?;
        let score = py.allow_threads(|| trainer.training_score().to_vec());
        Ok(numpy::PyArray1::from_vec(py, score))
    }

    fn update_custom(
        &mut self,
        py: Python<'_>,
        gradients: Vec<f32>,
        hessians: Vec<f32>,
    ) -> PyResult<bool> {
        let trainer = self
            .trainer
            .as_mut()
            .ok_or_else(|| LightGBMError::new_err("Booster has no training Dataset"))?;
        let finished = py
            .allow_threads(|| trainer.update_custom(&gradients, &hessians))
            .map_err(to_py)?;
        self.model = trainer.snapshot();
        Ok(finished)
    }

    fn reset_learning_rate(&mut self, learning_rate: f64) -> PyResult<()> {
        let Some(trainer) = self.trainer.as_mut() else {
            return Err(LightGBMError::new_err(
                "Cannot reset parameters on a Booster loaded from a model.",
            ));
        };
        trainer.reset_learning_rate(learning_rate).map_err(to_py)
    }

    fn eval(
        &self,
        py: Python<'_>,
        dataset: PyRef<'_, NativeDataset>,
        params: HashMap<String, String>,
        data_name: String,
    ) -> PyResult<Vec<(String, String, f64, bool)>> {
        let model = &self.model;
        let ds = Arc::clone(&dataset.ds);
        let data = Arc::clone(&dataset.data);
        let (nrow, ncol) = (dataset.nrow, dataset.ncol);
        py.allow_threads(|| -> lgbm_core::Result<Vec<(String, String, f64, bool)>> {
            let cfg = Config::from_params(&params)?;
            let pred_cfg = PredictConfig {
                kind: PredictKind::RawScore,
                num_threads: cfg.num_threads.max(0) as usize,
                ..PredictConfig::default()
            };
            let predictor = Predictor::new(model, &pred_cfg)?;
            let row_scores = data.predict(&predictor, nrow, ncol)?;
            let nclass = model.num_class();
            let scores = if nclass > 1 {
                let mut class_major = vec![0.0; row_scores.len()];
                for row in 0..nrow {
                    for class in 0..nclass {
                        class_major[class * nrow + row] = row_scores[row * nclass + class];
                    }
                }
                class_major
            } else {
                row_scores
            };
            // upstream: the validation ScoreUpdater starts from the init score.
            let mut scores = scores;
            if !ds.metadata.init_score.is_empty() {
                if ds.metadata.init_score.len() != scores.len() {
                    return Err(lgbm_core::Error::InvalidParameter(
                        "Number of class for initial score error".into(),
                    ));
                }
                for (score, init) in scores.iter_mut().zip(&ds.metadata.init_score) {
                    *score += init;
                }
            }
            let mut results = Vec::new();
            for metric_name in &cfg.metric {
                if let Some(metric) = create_metric(metric_name, &cfg, &ds.metadata)? {
                    let values = metric.eval(&scores, model.objective());
                    for (name, value) in metric.names().iter().zip(values) {
                        results.push((
                            data_name.clone(),
                            name.clone(),
                            value,
                            metric.factor_to_bigger_better() > 0.0,
                        ));
                    }
                }
            }
            Ok(results)
        })
        .map_err(to_py)
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
        self.check_num_features(ncol, disable_shape_check)?;
        let cfg = predict_config(
            kind,
            start_iteration,
            num_iteration,
            pred_early_stop,
            pred_early_stop_freq,
            pred_early_stop_margin,
            num_threads,
        )?;
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

    /// Predict a scipy CSR (`is_csc=false`) or CSC (`is_csc=true`) matrix of shape
    /// `nrow x ncol` given its `indptr` (int64), `indices` (int32) and `data` (float64).
    ///
    /// Returns a `nrow x num_pred_one_row` float64 array, except for `kind == 3` (contrib),
    /// which returns one `(indptr, indices, data)` triple per class in the input's format
    /// with `ncol + 1` columns, like upstream's `LGBM_BoosterPredictSparseOutput`.
    #[pyo3(signature = (indptr, indices, data, nrow, ncol, is_csc, kind, start_iteration,
        num_iteration, pred_early_stop=false, pred_early_stop_freq=10,
        pred_early_stop_margin=10.0, num_threads=0, disable_shape_check=false))]
    #[allow(clippy::too_many_arguments)]
    fn predict_sparse<'py>(
        &self,
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i32>,
        data: PyReadonlyArray1<'py, f64>,
        nrow: usize,
        ncol: usize,
        is_csc: bool,
        kind: i32,
        start_iteration: i32,
        num_iteration: i32,
        pred_early_stop: bool,
        pred_early_stop_freq: i32,
        pred_early_stop_margin: f64,
        num_threads: i32,
        disable_shape_check: bool,
    ) -> PyResult<Py<PyAny>> {
        self.check_num_features(ncol, disable_shape_check)?;
        let cfg = predict_config(
            kind,
            start_iteration,
            num_iteration,
            pred_early_stop,
            pred_early_stop_freq,
            pred_early_stop_margin,
            num_threads,
        )?;
        let as_slice = |e: numpy::NotContiguousError| LightGBMError::new_err(e.to_string());
        let (indptr, indices, data) = (
            indptr.as_slice().map_err(as_slice)?,
            indices.as_slice().map_err(as_slice)?,
            data.as_slice().map_err(as_slice)?,
        );
        let model = &self.model;
        if cfg.kind == PredictKind::Contrib {
            let mats = py
                .allow_threads(|| {
                    let p = Predictor::new(model, &cfg)?;
                    if is_csc {
                        p.predict_contrib_csc(indptr, indices, data, nrow)
                    } else {
                        p.predict_contrib_csr(indptr, indices, data)
                    }
                })
                .map_err(to_py)?;
            let out: Vec<Py<PyAny>> = mats
                .into_iter()
                .map(|m| {
                    let t = (
                        numpy::PyArray1::from_vec(py, m.indptr),
                        numpy::PyArray1::from_vec(py, m.indices),
                        numpy::PyArray1::from_vec(py, m.data),
                    );
                    Ok(t.into_pyobject(py)?.into_any().unbind())
                })
                .collect::<PyResult<_>>()?;
            return Ok(pyo3::types::PyList::new(py, out)?.into_any().unbind());
        }
        let (out, width) = py
            .allow_threads(|| {
                let p = Predictor::new(model, &cfg)?;
                let w = p.num_pred_one_row();
                let out = if is_csc {
                    p.predict_csc(indptr, indices, data, nrow)?
                } else {
                    p.predict_csr(indptr, indices, data)?
                };
                Ok::<_, lgbm_core::Error>((out, w))
            })
            .map_err(to_py)?;
        let rows = out.len().checked_div(width).unwrap_or(nrow);
        Ok(numpy::PyArray1::from_vec(py, out)
            .reshape([rows, width])?
            .into_any()
            .unbind())
    }

    /// upstream: src/c_api.cpp (v4.7.0) Booster::CreatePredictor shape check.
    fn check_num_features(&self, ncol: usize, disable_shape_check: bool) -> PyResult<()> {
        if !disable_shape_check && ncol != self.model.num_features() {
            return Err(LightGBMError::new_err(format!(
                "The number of features in data ({ncol}) is not the same as it was in training data ({}).\n\
                 You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.",
                self.model.num_features()
            )));
        }
        Ok(())
    }
}

/// Attach optional weights, query sizes and init scores to a freshly built dataset.
/// Names the columns (upstream: the `feature_name` of `Dataset`, used as the model's feature
/// names); `None` keeps the automatic `Column_i` names.
fn apply_feature_names(ds: &mut Dataset, names: Option<Vec<String>>) -> lgbm_core::Result<()> {
    if let Some(names) = names {
        if names.len() != ds.layout.num_total_features {
            return Err(lgbm_core::Error::InvalidParameter(format!(
                "Length of feature names ({}) doesn't equal with num_features ({})",
                names.len(),
                ds.layout.num_total_features
            )));
        }
        ds.feature_names = names;
    }
    Ok(())
}

fn with_metadata(
    mut ds: Dataset,
    weight: Option<Vec<f32>>,
    group: Option<Vec<i32>>,
    init_score: Option<Vec<f64>>,
) -> lgbm_core::Result<Dataset> {
    if let Some(w) = weight {
        ds.set_weights(&w)?;
    }
    if let Some(g) = group {
        ds.set_group(&g)?;
    }
    if let Some(init) = init_score {
        ds.set_init_score(&init)?;
    }
    Ok(ds)
}

/// Build a [`PredictConfig`] from the `C_API_PREDICT_*` code and prediction parameters.
fn predict_config(
    kind: i32,
    start_iteration: i32,
    num_iteration: i32,
    pred_early_stop: bool,
    pred_early_stop_freq: i32,
    pred_early_stop_margin: f64,
    num_threads: i32,
) -> PyResult<PredictConfig> {
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
    Ok(PredictConfig {
        kind,
        start_iteration,
        num_iteration,
        pred_early_stop,
        pred_early_stop_freq,
        pred_early_stop_margin,
        num_threads: usize::try_from(num_threads).unwrap_or(0),
    })
}

/// Every parameter's aliases (excluding the parameter itself), from the table generated
/// from upstream's `config.h` (upstream `Config::parameter2aliases`).
/// `load_text_file` result: features, labels, weight, group sizes and feature names.
type TextFileArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Vec<f64>,
    Option<Vec<f32>>,
    Option<Vec<i32>>,
    Vec<String>,
);

/// Load a text data file (upstream `DatasetLoader::LoadFromFile`, in-memory path): the dense
/// features, labels, the weight and group columns when requested, and the header names.
#[pyfunction]
#[pyo3(signature = (path, header, label_column, weight_column, group_column, ignore_column,
    precise_float_parser, num_features=0))]
#[allow(clippy::too_many_arguments)] // Python `load_text_file` arguments.
fn load_text_file<'py>(
    py: Python<'py>,
    path: String,
    header: bool,
    label_column: String,
    weight_column: String,
    group_column: String,
    ignore_column: String,
    precise_float_parser: bool,
    num_features: usize,
) -> PyResult<TextFileArrays<'py>> {
    let content = std::fs::read_to_string(&path)
        .map_err(|_| LightGBMError::new_err(format!("Data file {path} doesn't exist.")))?;
    let opts = lgbm_core::io::text_loader::TextLoadOptions {
        header,
        label_column: &label_column,
        weight_column: &weight_column,
        group_column: &group_column,
        ignore_column: &ignore_column,
        precise_float_parser,
        num_features,
    };
    let loaded = py
        .allow_threads(|| lgbm_core::io::text_loader::load_text(&content, &opts))
        .map_err(to_py)?;
    let matrix = numpy::PyArray1::from_vec(py, loaded.features)
        .reshape([loaded.num_data, loaded.num_features])?;
    Ok((
        matrix,
        loaded.label,
        loaded.weight,
        loaded.group,
        loaded.feature_names,
    ))
}

/// Parse the text of a `<data>.weight` side file (upstream `Metadata::LoadWeights`).
#[pyfunction]
fn parse_weights_file(content: &str) -> PyResult<Vec<f32>> {
    lgbm_core::io::side_files::parse_weights(content).map_err(to_py)
}

/// Parse the text of a `<data>.query` side file into group sizes (upstream `LoadQueryBoundaries`).
#[pyfunction]
fn parse_query_file(content: &str) -> PyResult<Vec<i32>> {
    lgbm_core::io::side_files::parse_query(content).map_err(to_py)
}

/// Parse the text of a `<data>.init` side file: `(num_class, class-major values)`
/// (upstream `Metadata::LoadInitialScore`).
#[pyfunction]
fn parse_init_score_file(content: &str) -> PyResult<(usize, Vec<f64>)> {
    lgbm_core::io::side_files::parse_init_score(content).map_err(to_py)
}

#[pyfunction]
fn param_aliases() -> HashMap<&'static str, Vec<&'static str>> {
    lgbm_core::config::PARAMETER_TO_ALIASES
        .iter()
        .map(|(name, aliases)| (*name, aliases.to_vec()))
        .collect()
}

/// Every parameter's type name (`bool`, `int`, `double`, `string`, `vector<...>`), from the table
/// generated from upstream's `config.h` (upstream `Config::ParameterTypes`).
#[pyfunction]
fn param_types() -> HashMap<&'static str, &'static str> {
    lgbm_core::config::PARAMETER_TYPES.iter().copied().collect()
}

/// Forwards the core crate's `log` records (parameter warnings) to the Python logger, as
/// upstream's log callback does. Routed through `bankai_lightgbm.basic._native_log`.
struct PyLogger;

impl log::Log for PyLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) || CONFIG_LOGS_MUTED.load(Ordering::Relaxed) {
            return;
        }
        let level = if record.level() == log::Level::Warn {
            "warning"
        } else {
            "info"
        };
        let message = record.args().to_string();
        // Logging must never fail the caller: a missing Python side only drops the message.
        let _ = Python::with_gil(|py| -> PyResult<()> {
            py.import("bankai_lightgbm.basic")?
                .call_method1("_native_log", (level, message))?;
            Ok(())
        });
    }

    fn flush(&self) {}
}

static PY_LOGGER: PyLogger = PyLogger;

/// Set while a config is re-parsed for a call that already reported its warnings (e.g.
/// `NativeBooster.train` after `validate_params` in the `Booster` constructor), so each
/// warning is printed once, as upstream does.
static CONFIG_LOGS_MUTED: AtomicBool = AtomicBool::new(false);

/// Restores the previous mute state when dropped.
struct MuteGuard(bool);

impl MuteGuard {
    fn new(mute: bool) -> Self {
        MuteGuard(CONFIG_LOGS_MUTED.swap(mute, Ordering::Relaxed))
    }
}

impl Drop for MuteGuard {
    fn drop(&mut self) {
        CONFIG_LOGS_MUTED.store(self.0, Ordering::Relaxed);
    }
}

/// Parse `params` the way `LGBM_BoosterCreate` does, raising on invalid values and emitting
/// the same config warnings (upstream constructs the booster in ``Booster.__init__``).
#[pyfunction]
#[pyo3(signature = (params))]
fn validate_params(params: HashMap<String, String>) -> PyResult<()> {
    Config::from_params(&params).map(|_| ()).map_err(to_py)
}

/// upstream: `LGBM_DatasetUpdateParamChecking`: raise when `new` cannot be applied to a Dataset built with `old`.
#[pyfunction]
#[pyo3(signature = (old_params, new_params))]
fn dataset_check_reset_config(
    old_params: HashMap<String, String>,
    new_params: HashMap<String, String>,
) -> PyResult<()> {
    let old = Config::from_params(&old_params).map_err(to_py)?;
    lgbm_core::config::check_dataset_reset_config(&old, &new_params).map_err(to_py)
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Install once per process; a second module init keeps the first logger.
    if log::set_logger(&PY_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    m.add("UPSTREAM_VERSION", lgbm_core::UPSTREAM_VERSION)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("LightGBMError", m.py().get_type::<LightGBMError>())?;
    m.add_class::<NativeBooster>()?;
    m.add_class::<NativeDataset>()?;
    m.add_function(wrap_pyfunction!(param_aliases, m)?)?;
    m.add_function(wrap_pyfunction!(load_text_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_weights_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_query_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_init_score_file, m)?)?;
    m.add_function(wrap_pyfunction!(param_types, m)?)?;
    m.add_function(wrap_pyfunction!(validate_params, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_check_reset_config, m)?)?;
    Ok(())
}
