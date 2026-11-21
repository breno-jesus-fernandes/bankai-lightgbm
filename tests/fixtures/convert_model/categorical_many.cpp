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
double PredictTree0(const double* arr) { const std::vector<uint32_t> cat_threshold = {2454267026,613564708,1};double fval = 0.0f; int int_fval = 0; fval = arr[1];if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }if (int_fval >= 0 && int_fval < 32 * (3) && (((cat_threshold[0 + int_fval / 32] >> (int_fval & 31)) & 1))) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.50482697977453261) {return 1.1623632259100718; } else { return 1.3461228263609681; } } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.4057581329230795) {return 0.87274634127056094; } else { return 1.0450830014513339; } } }
double PredictTree0ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {2454267026,613564708,1};double fval = 0.0f; int int_fval = 0; fval = arr.count(1) > 0 ? arr.at(1) : 0.0f;if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }if (int_fval >= 0 && int_fval < 32 * (3) && (((cat_threshold[0 + int_fval / 32] >> (int_fval & 31)) & 1))) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.50482697977453261) {return 1.1623632259100718; } else { return 1.3461228263609681; } } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.4057581329230795) {return 0.87274634127056094; } else { return 1.0450830014513339; } } }

double PredictTree1(const double* arr) { const std::vector<uint32_t> cat_threshold = {2454267026,613564708,1};double fval = 0.0f; int int_fval = 0; fval = arr[1];if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }if (int_fval >= 0 && int_fval < 32 * (3) && (((cat_threshold[0 + int_fval / 32] >> (int_fval & 31)) & 1))) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.14141851678767722) {return 0.10789347158537971; } else { return 0.27549163265661764; } } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.099984955440577464) {return -0.14834806311805293; } else { return -0.0040128812193870548; } } }
double PredictTree1ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {2454267026,613564708,1};double fval = 0.0f; int int_fval = 0; fval = arr.count(1) > 0 ? arr.at(1) : 0.0f;if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }if (int_fval >= 0 && int_fval < 32 * (3) && (((cat_threshold[0 + int_fval / 32] >> (int_fval & 31)) & 1))) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.14141851678767722) {return 0.10789347158537971; } else { return 0.27549163265661764; } } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.099984955440577464) {return -0.14834806311805293; } else { return -0.0040128812193870548; } } }

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

double PredictTree0Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {2454267026,613564708,1};double fval = 0.0f; int int_fval = 0; fval = arr[1];if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }if (int_fval >= 0 && int_fval < 32 * (3) && (((cat_threshold[0 + int_fval / 32] >> (int_fval & 31)) & 1))) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.50482697977453261) {return 0; } else { return 3; } } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.4057581329230795) {return 1; } else { return 2; } } }
double PredictTree0LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {2454267026,613564708,1};double fval = 0.0f; int int_fval = 0; fval = arr.count(1) > 0 ? arr.at(1) : 0.0f;if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }if (int_fval >= 0 && int_fval < 32 * (3) && (((cat_threshold[0 + int_fval / 32] >> (int_fval & 31)) & 1))) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.50482697977453261) {return 0; } else { return 3; } } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.4057581329230795) {return 1; } else { return 2; } } }

double PredictTree1Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {2454267026,613564708,1};double fval = 0.0f; int int_fval = 0; fval = arr[1];if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }if (int_fval >= 0 && int_fval < 32 * (3) && (((cat_threshold[0 + int_fval / 32] >> (int_fval & 31)) & 1))) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.14141851678767722) {return 0; } else { return 3; } } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.099984955440577464) {return 1; } else { return 2; } } }
double PredictTree1LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {2454267026,613564708,1};double fval = 0.0f; int int_fval = 0; fval = arr.count(1) > 0 ? arr.at(1) : 0.0f;if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }if (int_fval >= 0 && int_fval < 32 * (3) && (((cat_threshold[0 + int_fval / 32] >> (int_fval & 31)) & 1))) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.14141851678767722) {return 0; } else { return 3; } } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.099984955440577464) {return 1; } else { return 2; } } }

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
