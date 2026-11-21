#include "gbdt.h"
#include <LightGBM/utils/common.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/prediction_early_stop.h>
#include <ctime>
#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <utility>
namespace LightGBM {
double PredictTree0(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) {fval = arr[0];if (!std::isnan(fval)) {return -0.091009920020587748; } else { fval = arr[1];if (!std::isnan(fval)) {return 0.25664613066732678; } else { return 0.090616834188187306; } } } else { return 0.40088619109460966; } }
double PredictTree0ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (!std::isnan(fval)) {return -0.091009920020587748; } else { fval = arr.count(1) > 0 ? arr.at(1) : 0.0f;if (!std::isnan(fval)) {return 0.25664613066732678; } else { return 0.090616834188187306; } } } else { return 0.40088619109460966; } }

double PredictTree1(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) {fval = arr[0];if (!std::isnan(fval)) {return -0.2495204350222712; } else { fval = arr[1];if (!std::isnan(fval)) {return 0.10217618942260742; } else { return -0.067135272912967656; } } } else { return 0.20527527328278566; } }
double PredictTree1ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (!std::isnan(fval)) {return -0.2495204350222712; } else { fval = arr.count(1) > 0 ? arr.at(1) : 0.0f;if (!std::isnan(fval)) {return 0.10217618942260742; } else { return -0.067135272912967656; } } } else { return 0.20527527328278566; } }

double (*PredictTreePtr[])(const double*) = { PredictTree0 , PredictTree1 };

void GBDT::PredictRaw(const double* features, double *output, const PredictionEarlyStopInstance* early_stop) const {
	int early_stop_round_counter = 0;
	std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);
	for (int i = 0; i < num_iteration_for_pred_; ++i) {
		for (int k = 0; k < num_tree_per_iteration_; ++k) {
			output[k] += (*PredictTreePtr[i * num_tree_per_iteration_ + k])(features);
		}
		++early_stop_round_counter;
		if (early_stop->round_period == early_stop_round_counter) {
			if (early_stop->callback_function(output, num_tree_per_iteration_))
				return;
			early_stop_round_counter = 0;
		}
	}
}

double (*PredictTreeByMapPtr[])(const std::unordered_map<int, double>&) = { PredictTree0ByMap , PredictTree1ByMap };

void GBDT::PredictRawByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {
	int early_stop_round_counter = 0;
	std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);
	for (int i = 0; i < num_iteration_for_pred_; ++i) {
		for (int k = 0; k < num_tree_per_iteration_; ++k) {
			output[k] += (*PredictTreeByMapPtr[i * num_tree_per_iteration_ + k])(features);
		}
		++early_stop_round_counter;
		if (early_stop->round_period == early_stop_round_counter) {
			if (early_stop->callback_function(output, num_tree_per_iteration_))
				return;
			early_stop_round_counter = 0;
		}
	}
}

void GBDT::Predict(const double* features, double *output, const PredictionEarlyStopInstance* early_stop) const {
	PredictRaw(features, output, early_stop);
	if (average_output_) {
		for (int k = 0; k < num_tree_per_iteration_; ++k) {
			output[k] /= num_iteration_for_pred_;
		}
	}
	if (objective_function_ != nullptr) {
		objective_function_->ConvertOutput(output, output);
	}
}

void GBDT::PredictByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {
	PredictRawByMap(features, output, early_stop);
	if (average_output_) {
		for (int k = 0; k < num_tree_per_iteration_; ++k) {
			output[k] /= num_iteration_for_pred_;
		}
	}
	if (objective_function_ != nullptr) {
		objective_function_->ConvertOutput(output, output);
	}
}

double PredictTree0Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) {fval = arr[0];if (!std::isnan(fval)) {return 0; } else { fval = arr[1];if (!std::isnan(fval)) {return 2; } else { return 3; } } } else { return 1; } }
double PredictTree0LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (!std::isnan(fval)) {return 0; } else { fval = arr.count(1) > 0 ? arr.at(1) : 0.0f;if (!std::isnan(fval)) {return 2; } else { return 3; } } } else { return 1; } }

double PredictTree1Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) {fval = arr[0];if (!std::isnan(fval)) {return 0; } else { fval = arr[1];if (!std::isnan(fval)) {return 2; } else { return 3; } } } else { return 1; } }
double PredictTree1LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (!std::isnan(fval)) {return 0; } else { fval = arr.count(1) > 0 ? arr.at(1) : 0.0f;if (!std::isnan(fval)) {return 2; } else { return 3; } } } else { return 1; } }

double (*PredictTreeLeafPtr[])(const double*) = { PredictTree0Leaf , PredictTree1Leaf };

void GBDT::PredictLeafIndex(const double* features, double *output) const {
	int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;
	for (int i = 0; i < total_tree; ++i) {
		output[i] = (*PredictTreeLeafPtr[i])(features);
	}
}
double (*PredictTreeLeafByMapPtr[])(const std::unordered_map<int, double>&) = { PredictTree0LeafByMap , PredictTree1LeafByMap };

void GBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const {
	int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;
	for (int i = 0; i < total_tree; ++i) {
		output[i] = (*PredictTreeLeafByMapPtr[i])(features);
	}
}
}  // namespace LightGBM
