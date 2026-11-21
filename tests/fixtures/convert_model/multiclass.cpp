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
double PredictTree0(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.70875032260677517) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -1.3948724288726961) {return -1.284348767602705; } else { fval = arr[2];if (std::isnan(fval)) fval = 0.0;if (fval <= 1.173227494597781) {return -1.0697917890007824; } else { return -1.2843487676027048; } } } else { return -1.2056778754486668; } }
double PredictTree0ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.70875032260677517) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -1.3948724288726961) {return -1.284348767602705; } else { fval = arr.count(2) > 0 ? arr.at(2) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 1.173227494597781) {return -1.0697917890007824; } else { return -1.2843487676027048; } } } else { return -1.2056778754486668; } }

double PredictTree1(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 1.2061452258833569) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.92167051195709704) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.7879991525025466) {return -1.026543538726012; } else { return -0.96024445769008882; } } else { return -0.80305147523394838; } } else { return -1.0425836389766385; } }
double PredictTree1ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 1.2061452258833569) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.92167051195709704) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.7879991525025466) {return -1.026543538726012; } else { return -0.96024445769008882; } } else { return -0.80305147523394838; } } else { return -1.0425836389766385; } }

double PredictTree2(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -1.3948724288726961) {return -0.99157099041220687; } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.096832834280637048) {return -1.178467348793151; } else { fval = arr[2];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.64726961408956896) {return -1.1894325849432714; } else { return -1.0952127780237169; } } } }
double PredictTree2ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -1.3948724288726961) {return -0.99157099041220687; } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.096832834280637048) {return -1.178467348793151; } else { fval = arr.count(2) > 0 ? arr.at(2) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.64726961408956896) {return -1.1894325849432714; } else { return -1.0952127780237169; } } } }

double PredictTree3(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.70875032260677517) {fval = arr[2];if (std::isnan(fval)) fval = 0.0;if (fval <= 1.173227494597781) {return 0.06939449302855924; } else { return -0.094183051315955016; } } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 1.503084872255559) {return -0.029744848839476756; } else { return 0.073075220768706123; } } }
double PredictTree3ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.70875032260677517) {fval = arr.count(2) > 0 ? arr.at(2) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 1.173227494597781) {return 0.06939449302855924; } else { return -0.094183051315955016; } } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 1.503084872255559) {return -0.029744848839476756; } else { return 0.073075220768706123; } } }

double PredictTree4(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 1.2061452258833569) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.92167051195709704) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.7879991525025466) {return -0.041884527383295747; } else { return 0.018551966606511559; } } else { return 0.15921531406875286; } } else { return -0.058426545597130314; } }
double PredictTree4ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 1.2061452258833569) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.92167051195709704) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.7879991525025466) {return -0.041884527383295747; } else { return 0.018551966606511559; } } else { return 0.15921531406875286; } } else { return -0.058426545597130314; } }

double PredictTree5(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -1.2483753173716234) {return 0.10548374490728779; } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.096832834280637048) {return -0.040783001197743446; } else { fval = arr[2];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.80355546258306798) {return 0.036744474545224393; } else { return -0.066272820522636514; } } } }
double PredictTree5ByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -1.2483753173716234) {return 0.10548374490728779; } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.096832834280637048) {return -0.040783001197743446; } else { fval = arr.count(2) > 0 ? arr.at(2) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.80355546258306798) {return 0.036744474545224393; } else { return -0.066272820522636514; } } } }

double (*PredictTreePtr[])(const double*) = { PredictTree0 , PredictTree1 , PredictTree2 , PredictTree3 , PredictTree4 , PredictTree5 };

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

double (*PredictTreeByMapPtr[])(const std::unordered_map<int, double>&) = { PredictTree0ByMap , PredictTree1ByMap , PredictTree2ByMap , PredictTree3ByMap , PredictTree4ByMap , PredictTree5ByMap };

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

double PredictTree0Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.70875032260677517) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -1.3948724288726961) {return 0; } else { fval = arr[2];if (std::isnan(fval)) fval = 0.0;if (fval <= 1.173227494597781) {return 2; } else { return 3; } } } else { return 1; } }
double PredictTree0LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.70875032260677517) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -1.3948724288726961) {return 0; } else { fval = arr.count(2) > 0 ? arr.at(2) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 1.173227494597781) {return 2; } else { return 3; } } } else { return 1; } }

double PredictTree1Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 1.2061452258833569) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.92167051195709704) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.7879991525025466) {return 0; } else { return 3; } } else { return 2; } } else { return 1; } }
double PredictTree1LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 1.2061452258833569) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.92167051195709704) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.7879991525025466) {return 0; } else { return 3; } } else { return 2; } } else { return 1; } }

double PredictTree2Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -1.3948724288726961) {return 0; } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.096832834280637048) {return 1; } else { fval = arr[2];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.64726961408956896) {return 2; } else { return 3; } } } }
double PredictTree2LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -1.3948724288726961) {return 0; } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.096832834280637048) {return 1; } else { fval = arr.count(2) > 0 ? arr.at(2) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.64726961408956896) {return 2; } else { return 3; } } } }

double PredictTree3Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.70875032260677517) {fval = arr[2];if (std::isnan(fval)) fval = 0.0;if (fval <= 1.173227494597781) {return 0; } else { return 3; } } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 1.503084872255559) {return 1; } else { return 2; } } }
double PredictTree3LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.70875032260677517) {fval = arr.count(2) > 0 ? arr.at(2) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 1.173227494597781) {return 0; } else { return 3; } } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 1.503084872255559) {return 1; } else { return 2; } } }

double PredictTree4Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 1.2061452258833569) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.92167051195709704) {fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.7879991525025466) {return 0; } else { return 3; } } else { return 2; } } else { return 1; } }
double PredictTree4LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 1.2061452258833569) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.92167051195709704) {fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.7879991525025466) {return 0; } else { return 3; } } else { return 2; } } else { return 1; } }

double PredictTree5Leaf(const double* arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -1.2483753173716234) {return 0; } else { fval = arr[0];if (std::isnan(fval)) fval = 0.0;if (fval <= -0.096832834280637048) {return 1; } else { fval = arr[2];if (std::isnan(fval)) fval = 0.0;if (fval <= 0.80355546258306798) {return 2; } else { return 3; } } } }
double PredictTree5LeafByMap(const std::unordered_map<int, double>& arr) { const std::vector<uint32_t> cat_threshold = {};double fval = 0.0f; fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -1.2483753173716234) {return 0; } else { fval = arr.count(0) > 0 ? arr.at(0) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= -0.096832834280637048) {return 1; } else { fval = arr.count(2) > 0 ? arr.at(2) : 0.0f;if (std::isnan(fval)) fval = 0.0;if (fval <= 0.80355546258306798) {return 2; } else { return 3; } } } }

double (*PredictTreeLeafPtr[])(const double*) = { PredictTree0Leaf , PredictTree1Leaf , PredictTree2Leaf , PredictTree3Leaf , PredictTree4Leaf , PredictTree5Leaf };

void GBDT::PredictLeafIndex(const double* features, double *output) const {
	int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;
	for (int i = 0; i < total_tree; ++i) {
		output[i] = (*PredictTreeLeafPtr[i])(features);
	}
}
double (*PredictTreeLeafByMapPtr[])(const std::unordered_map<int, double>&) = { PredictTree0LeafByMap , PredictTree1LeafByMap , PredictTree2LeafByMap , PredictTree3LeafByMap , PredictTree4LeafByMap , PredictTree5LeafByMap };

void GBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const {
	int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;
	for (int i = 0; i < total_tree; ++i) {
		output[i] = (*PredictTreeLeafByMapPtr[i])(features);
	}
}
}  // namespace LightGBM
