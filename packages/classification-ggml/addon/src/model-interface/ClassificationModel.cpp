#include "ClassificationModel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>
#include <qvac-lib-inference-addon-cpp/Errors.hpp>
#include <qvac-lib-inference-addon-cpp/Logger.hpp>

#include "ImagePreprocessor.hpp"
#include "MobileNetGraph.hpp"

namespace qvac_lib_infer_ggml_classification {

using qvac_errors::StatusError;
using qvac_errors::general_error::InternalError;
using qvac_errors::general_error::InvalidArgument;

namespace {
constexpr const char* kModelName = "mobilenetv3-small-ggml-classification";
constexpr size_t INPUT_EDGE_SAMPLE_COUNT = 20;
}

ClassificationModel::ClassificationModel(std::string modelPath)
    : modelPath_(std::move(modelPath)) {}

ClassificationModel::~ClassificationModel() {
  // ggml requires buffers to be freed strictly before the backend they were
  // allocated on. Explicitly reset the compute graph and weights bundle (both
  // own backend-allocated buffers) before releasing the backend itself.
  compute_.reset();
  weights_.reset();
  if (backend_ != nullptr) {
    ggml_backend_free(backend_);
    backend_ = nullptr;
  }
}

std::string ClassificationModel::getName() const {
  return kModelName;
}

qvac_lib_inference_addon_cpp::RuntimeStats
ClassificationModel::runtimeStats() const {
  using qvac_lib_inference_addon_cpp::RuntimeStats;
  RuntimeStats stats;
  const double totalMs = static_cast<double>(lastInferenceUs_) / 1000.0;
  stats.emplace_back("total_time_ms", totalMs);
  return stats;
}

void ClassificationModel::setNumThreads(int threads) {
  std::scoped_lock lock(mutex_);
  numThreads_ = threads;
}

ClassifyOutput ClassificationModel::runTensor(std::span<const float> inputTensor) {
  ggml_backend_tensor_set(
      compute_.input, inputTensor.data(), 0,
      inputTensor.size() * sizeof(float));

  if (numThreads_ > 0) {
    ggml_backend_cpu_set_n_threads(backend_, numThreads_);
  }

  ggml_status status =
      ggml_backend_graph_compute(backend_, compute_.graph);
  if (status != GGML_STATUS_SUCCESS) {
    throw StatusError(
        InternalError, "ggml_backend_graph_compute failed with status " +
                           std::to_string(static_cast<int>(status)));
  }

  ClassifyOutput output;
  output.data_4.resize(ggml_nelements(compute_.output_4));
  ggml_backend_tensor_get(
      compute_.output_4, output.data_4.data(), 0, ggml_nbytes(compute_.output_4));

  return output;
}

namespace {

/// Numerically stable softmax over a short logits vector. Defensive
/// against non-finite inputs: if every logit is NaN/Inf we return a
/// uniform distribution rather than propagating the non-finite value;
/// if the exponential sum degenerates to zero or non-finite we also
/// fall back to uniform. The caller therefore always receives a
/// well-formed probability vector that sums to 1 in exactly one of
/// two ways (computed softmax, or uniform fallback).
std::vector<float> softmax(std::span<const float> logits) {
  if (logits.empty()) {
    return {};
  }

  // max_element on a span containing NaN is unspecified; walk by hand
  // and skip non-finite values so maxLogit stays finite whenever at
  // least one input is finite.
  float maxLogit = -std::numeric_limits<float>::infinity();
  for (const float logit : logits) {
    if (std::isfinite(logit) && logit > maxLogit) {
      maxLogit = logit;
    }
  }
  if (!std::isfinite(maxLogit)) {
    const float uniform = 1.0F / static_cast<float>(logits.size());
    return std::vector<float>(logits.size(), uniform);
  }

  std::vector<float> probs(logits.size());
  float sum = 0.0F;
  for (size_t i = 0; i < logits.size(); ++i) {
    const float diff = logits[i] - maxLogit;
    const float e = std::isfinite(diff) ? std::exp(diff) : 0.0F;
    probs[i] = e;
    sum += e;
  }

  if (std::isfinite(sum) && sum > 0.0F) {
    const float inv = 1.0F / sum;
    for (float& p : probs) {
      p *= inv;
    }
  } else {
    // Every diff saturated to -inf or the sum itself overflowed;
    // fall back to a uniform distribution so downstream code always
    // sees a valid probability vector.
    const float uniform = 1.0F / static_cast<float>(logits.size());
    std::fill(probs.begin(), probs.end(), uniform);
  }
  return probs;
}

/// Environment-variable-gated trace printer for per-inference
/// diagnostics. Off by default (no output). Turn on with
/// `QVAC_CLASSIFICATION_TRACE=1` to print raw logits, computed
/// probabilities, and sorted results to stderr. Invaluable for
/// debugging platform-specific numerical issues (e.g. the win32 CI
/// meal_1 anomaly) without changing the public logger wiring.
bool traceEnabled() {
  const char* v = std::getenv("QVAC_CLASSIFICATION_TRACE");
  return v != nullptr && v[0] == '1';
}

void printInputEdgeSamples(std::span<const float> values) {
  const size_t sampleCount = std::min(values.size(), INPUT_EDGE_SAMPLE_COUNT);

  std::cout << "  input first " << sampleCount << " elements:";
  for (const float value : values.first(sampleCount)) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';

  std::cout << "  input last " << sampleCount << " elements:";
  for (const float value : values.last(sampleCount)) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';
}

} // namespace

void ClassificationModel::load() {
  std::scoped_lock lock(mutex_);
  if (loaded_) {
    return;
  }
  if (modelPath_.empty()) {
    throw StatusError(
        InvalidArgument,
        "ClassificationModel requires a path to mobilenetv3 FP16 GGUF weights");
  }

  backend_ = ggml_backend_cpu_init();
  if (backend_ == nullptr) {
    throw StatusError(InternalError, "Failed to initialize ggml CPU backend");
  }

  labels_.clear();
  weights_ = graph::loadWeights(modelPath_, backend_, labels_);
  if (labels_.empty()) {
    labels_ = {"food", "report", "other"};
  }
  compute_ = graph::buildGraph(weights_, backend_);

  loaded_ = true;
}

std::any ClassificationModel::process(const std::any& input) {
  std::scoped_lock lock(mutex_);

  const auto* inPtr = std::any_cast<ClassifyInput>(&input);
  if (inPtr == nullptr) {
    throw StatusError(InvalidArgument, "ClassificationModel: invalid input type");
  }
  if (!loaded_) {
    throw StatusError(
        InternalError,
        "ClassificationModel: classify() called before load() or after unload()");
  }

  const auto t0 = std::chrono::steady_clock::now();

  // Preprocess: image buffer -> FP32 WHCN tensor (224x224x3). The
  // preprocessor still uses uint32_t = 0 internally as its
  // "encoded path" indicator, but the JS-facing ClassifyInput now
  // carries an explicit std::optional<RawRgbDims>; translate at the
  // boundary so the preprocessor signature stays cheap.
  const uint32_t rawW = inPtr->rawRgb.has_value() ? inPtr->rawRgb->width : 0;
  const uint32_t rawH = inPtr->rawRgb.has_value() ? inPtr->rawRgb->height : 0;
  const uint32_t rawC =
      inPtr->rawRgb.has_value() ? inPtr->rawRgb->channels : 0;
  std::vector<float> inputTensor = preprocess::preprocessToTensor(
      std::span<const uint8_t>(inPtr->data.data(), inPtr->data.size()),
      rawW, rawH, rawC);

  const size_t expected = static_cast<size_t>(preprocess::kInputSize) *
                          preprocess::kInputSize * preprocess::kChannels;
  if (inputTensor.size() != expected) {
    throw StatusError(
        InternalError, "ClassificationModel: preprocessed tensor has wrong size");
  }

  printInputEdgeSamples(inputTensor);
  ggml_backend_tensor_set(
      compute_.input, inputTensor.data(), 0,
      inputTensor.size() * sizeof(float));

  // Configure CPU threads if requested; otherwise libggml picks a sensible
  // default based on hardware concurrency.
  if (numThreads_ > 0) {
    ggml_backend_cpu_set_n_threads(backend_, numThreads_);
  }

  ggml_status status =
      ggml_backend_graph_compute(backend_, compute_.graph);
  if (status != GGML_STATUS_SUCCESS) {
    throw StatusError(
        InternalError, "ggml_backend_graph_compute failed with status " +
                           std::to_string(static_cast<int>(status)));
  }

  // Retrieve logits.
  ClassifyOutput output;
  output.data_4.resize(ggml_nelements(compute_.output_4));
  ggml_backend_tensor_get(
      compute_.output_4, output.data_4.data(), 0, ggml_nbytes(compute_.output_4));

  return std::any(std::move(output));
}

} // namespace qvac_lib_infer_ggml_classification

