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
double PredictTree0(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.59749190179185907) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.53850746151840856) {return -0.079568483187253375; } else { return 0.12026418784939154; } } else { fval = arr[1];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.31279456603191025) {return 0.49728280258132146; } else { return 0.31858173833157788; } } }
double PredictTree0ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.59749190179185907) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.53850746151840856) {return -0.079568483187253375; } else { return 0.12026418784939154; } } else { fval = arr.count(1) > 0 ? arr.at(1) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.31279456603191025) {return 0.49728280258132146; } else { return 0.31858173833157788; } } }

double PredictTree1(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.26654623221443224) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.81335691177910174) {return -0.23209392748497151; } else { return -0.053384359512064196; } } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.93400159897469737) {return 0.11497651617974043; } else { return 0.27781978042229366; } } }
double PredictTree1ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.26654623221443224) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.81335691177910174) {return -0.23209392748497151; } else { return -0.053384359512064196; } } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.93400159897469737) {return 0.11497651617974043; } else { return 0.27781978042229366; } } }

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

double PredictTree0Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.59749190179185907) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.53850746151840856) {return 0; } else { return 2; } } else { fval = arr[1];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.31279456603191025) {return 1; } else { return 3; } } }
double PredictTree0LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.59749190179185907) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.53850746151840856) {return 0; } else { return 2; } } else { fval = arr.count(1) > 0 ? arr.at(1) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.31279456603191025) {return 1; } else { return 3; } } }

double PredictTree1Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.26654623221443224) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.81335691177910174) {return 0; } else { return 2; } } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.93400159897469737) {return 1; } else { return 3; } } }
double PredictTree1LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.26654623221443224) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.81335691177910174) {return 0; } else { return 2; } } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.93400159897469737) {return 1; } else { return 3; } } }

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
