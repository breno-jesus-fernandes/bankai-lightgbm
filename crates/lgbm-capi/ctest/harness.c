/*
 * C-header harness for `lgbm-capi` (ROADMAP T5.3 "Verify").
 *
 * Compiles against the *verbatim upstream* header `third_party/lightgbm/include/LightGBM/c_api.h`
 * (v4.7.0) and links the `lgbm-capi` static library, exercising the implemented `LGBM_*` symbols
 * end to end. Because the call sites are type-checked against the upstream prototypes and then run
 * for real, this catches both ABI/signature drift and link-time symbol gaps that the Rust
 * integration tests (which use the crate's own declarations) cannot. Driven by
 * `scripts/c_header_harness.sh`.
 */
#include <LightGBM/c_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, what)                                                        \
  do {                                                                           \
    if (!(cond)) {                                                               \
      fprintf(stderr, "c_api harness FAIL: %s | last error: %s\n", (what),       \
              LGBM_GetLastError());                                              \
      return 1;                                                                  \
    }                                                                            \
  } while (0)

int main(void) {
  const int nrow = 40;
  const int ncol = 4;
  double data[40 * 4];
  float label[40];
  for (int i = 0; i < nrow * ncol; i++) {
    data[i] = ((double)(i * 37 % 101)) * 0.1 - 5.0;
  }
  for (int r = 0; r < nrow; r++) {
    label[r] = (float)(data[r * ncol] * 2.0 - data[r * ncol + ncol - 1] + (r % 5) * 0.3);
  }

  const char *params =
      "objective=regression metric=l2,l1 max_bin=31 num_iterations=5 num_leaves=7 "
      "learning_rate=0.1 min_data_in_leaf=1 deterministic=true force_col_wise=true "
      "num_threads=1 seed=42 verbosity=-1";

  DatasetHandle ds = NULL;
  CHECK(LGBM_DatasetCreateFromMat(data, C_API_DTYPE_FLOAT64, nrow, ncol, 1, params,
                                  NULL, &ds) == 0,
        "LGBM_DatasetCreateFromMat");

  CHECK(LGBM_DatasetSetField(ds, "label", label, nrow, C_API_DTYPE_FLOAT32) == 0,
        "LGBM_DatasetSetField");

  int num_data = 0;
  CHECK(LGBM_DatasetGetNumData(ds, &num_data) == 0 && num_data == nrow,
        "LGBM_DatasetGetNumData");

  BoosterHandle booster = NULL;
  CHECK(LGBM_BoosterCreate(ds, params, &booster) == 0, "LGBM_BoosterCreate");

  for (int i = 0; i < 5; i++) {
    int empty = -1;
    CHECK(LGBM_BoosterUpdateOneIter(booster, &empty) == 0, "LGBM_BoosterUpdateOneIter");
  }

  /* Eval: counts, names (two-call string-array protocol), then values. */
  int eval_count = 0;
  CHECK(LGBM_BoosterGetEvalCounts(booster, &eval_count) == 0 && eval_count == 2,
        "LGBM_BoosterGetEvalCounts");

  int out_len = 0;
  size_t buffer_len = 0;
  CHECK(LGBM_BoosterGetEvalNames(booster, 0, &out_len, 0, &buffer_len, NULL) == 0 &&
            out_len == 2 && buffer_len > 0,
        "LGBM_BoosterGetEvalNames (size query)");
  char storage[2][64];
  char *names[2] = {storage[0], storage[1]};
  CHECK(LGBM_BoosterGetEvalNames(booster, 2, &out_len, sizeof(storage[0]), &buffer_len,
                                 names) == 0,
        "LGBM_BoosterGetEvalNames");
  CHECK(strcmp(names[0], "l2") == 0 && strcmp(names[1], "l1") == 0,
        "eval names are l2 then l1");

  double evals[2] = {-1.0, -1.0};
  CHECK(LGBM_BoosterGetEval(booster, 0, &out_len, evals) == 0 && out_len == 2,
        "LGBM_BoosterGetEval");
  CHECK(evals[0] >= 0.0 && evals[1] >= 0.0, "eval values are non-negative");

  /* Prediction: size then dense predict. */
  int64_t num_pred = 0;
  CHECK(LGBM_BoosterCalcNumPredict(booster, nrow, C_API_PREDICT_NORMAL, 0, 0, &num_pred) == 0 &&
            num_pred == nrow,
        "LGBM_BoosterCalcNumPredict");
  double *preds = (double *)malloc(sizeof(double) * (size_t)num_pred);
  CHECK(preds != NULL, "malloc predictions");
  int64_t pred_out_len = 0;
  CHECK(LGBM_BoosterPredictForMat(booster, data, C_API_DTYPE_FLOAT64, nrow, ncol, 1,
                                  C_API_PREDICT_NORMAL, 0, 0, "", &pred_out_len, preds) == 0 &&
            pred_out_len == nrow,
        "LGBM_BoosterPredictForMat");
  free(preds);

  /* Serialization: two-call buffer protocol. */
  int64_t needed = 0;
  CHECK(LGBM_BoosterSaveModelToString(booster, 0, 0, C_API_PREDICT_NORMAL, 0, &needed, NULL) == 0 &&
            needed > 0,
        "LGBM_BoosterSaveModelToString (size query)");
  char *model = (char *)malloc((size_t)needed);
  CHECK(model != NULL, "malloc model string");
  int64_t written = 0;
  CHECK(LGBM_BoosterSaveModelToString(booster, 0, 0, 0, needed, &written, model) == 0 &&
            written == needed,
        "LGBM_BoosterSaveModelToString");
  CHECK(strstr(model, "tree") != NULL, "model text starts a tree section");
  free(model);

  CHECK(LGBM_BoosterFree(booster) == 0, "LGBM_BoosterFree");
  CHECK(LGBM_DatasetFree(ds) == 0, "LGBM_DatasetFree");

  printf("c_api harness: OK (%d eval metrics, %d rows)\n", eval_count, nrow);
  return 0;
}
