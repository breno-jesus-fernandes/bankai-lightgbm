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
{{RAW_TREES}}double (*PredictTreePtr[])(const double*) = { {{RAW_POINTERS}} };

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

double (*PredictTreeByMapPtr[])(const std::unordered_map<int, double>&) = { {{MAP_POINTERS}} };

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

{{LEAF_TREES}}double (*PredictTreeLeafPtr[])(const double*) = { {{LEAF_POINTERS}} };

void GBDT::PredictLeafIndex(const double* features, double *output) const {
	int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;
	for (int i = 0; i < total_tree; ++i) {
		output[i] = (*PredictTreeLeafPtr[i])(features);
	}
}
double (*PredictTreeLeafByMapPtr[])(const std::unordered_map<int, double>&) = { {{LEAF_MAP_POINTERS}} };

void GBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const {
	int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;
	for (int i = 0; i < total_tree; ++i) {
		output[i] = (*PredictTreeLeafByMapPtr[i])(features);
	}
}
}  // namespace LightGBM
