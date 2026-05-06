#include <onnxruntime/onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifndef DEFAULT_DB_MOBILENET_ONNX_PATH
#define DEFAULT_DB_MOBILENET_ONNX_PATH "onnx_models/db_mobilenet_v3_large.onnx"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t DYNAMIC_BATCH = 1;
constexpr int64_t DYNAMIC_CHANNEL = 3;
constexpr int64_t DYNAMIC_SPATIAL = 1024;
constexpr uint32_t RANDOM_SEED = 12345;
constexpr double P50 = 50.0;
constexpr double P95 = 95.0;

struct Config {
  std::string modelPath{DEFAULT_DB_MOBILENET_ONNX_PATH};
  int warmupRuns{3};
  int benchmarkRuns{20};
};

void printUsage(std::string_view appName) {
  std::cout << "Usage: " << appName
            << " [model_path] [benchmark_runs] [warmup_runs]\n"
            << "Defaults:\n"
            << "  model_path: " << DEFAULT_DB_MOBILENET_ONNX_PATH << '\n'
            << "  benchmark_runs: 20\n"
            << "  warmup_runs: 3\n";
}

int parsePositiveInt(const std::string& value, std::string_view label) {
  const int parsed = std::stoi(value);
  if (parsed <= 0) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return parsed;
}

std::vector<std::string> argsFromArgv(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    args.emplace_back(argv[i]);
  }
  return args;
}

Config parseArgs(const std::vector<std::string>& args) {
  Config config;
  if (args.size() > 1) {
    const std::string& firstArg = args[1];
    if (firstArg == "-h" || firstArg == "--help") {
      printUsage(args[0]);
      std::exit(0);
    }
    config.modelPath = firstArg;
  }
  if (args.size() > 2) {
    config.benchmarkRuns = parsePositiveInt(args[2], "benchmark_runs");
  }
  if (args.size() > 3) {
    config.warmupRuns = parsePositiveInt(args[3], "warmup_runs");
  }
  if (args.size() > 4) {
    throw std::invalid_argument("too many arguments");
  }
  return config;
}

std::vector<int64_t> resolveInputShape(const std::vector<int64_t>& modelShape) {
  std::vector<int64_t> shape = modelShape;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > 0) {
      continue;
    }
    if (i == 0) {
      shape[i] = DYNAMIC_BATCH;
    } else if (i == 1) {
      shape[i] = DYNAMIC_CHANNEL;
    } else if (shape.size() >= 4 && i >= shape.size() - 2) {
      shape[i] = DYNAMIC_SPATIAL;
    } else {
      shape[i] = 1;
    }
  }
  return shape;
}

size_t elementCount(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         [](size_t total, int64_t dim) {
                           if (dim <= 0) {
                             throw std::runtime_error(
                                 "input shape contains unresolved dimension");
                           }
                           const auto dimSize = static_cast<size_t>(dim);
                           if (total > std::numeric_limits<size_t>::max() /
                                           dimSize) {
                             throw std::overflow_error(
                                 "input tensor element count overflow");
                           }
                           return total * dimSize;
                         });
}

void fillRandomInput(std::vector<float>& input) {
  std::mt19937 rng{RANDOM_SEED};
  std::uniform_real_distribution<float> distribution{0.0F, 1.0F};
  std::generate(input.begin(), input.end(),
                [&]() { return distribution(rng); });
}

std::vector<const char*> rawNames(const std::vector<std::string>& names) {
  std::vector<const char*> result;
  result.reserve(names.size());
  for (const auto& name : names) {
    result.push_back(name.c_str());
  }
  return result;
}

std::vector<std::string> getInputNames(Ort::Session& session,
                                       Ort::AllocatorWithDefaultOptions& allocator) {
  std::vector<std::string> names;
  const size_t count = session.GetInputCount();
  names.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    auto name = session.GetInputNameAllocated(i, allocator);
    names.emplace_back(name.get());
  }
  return names;
}

std::vector<std::string> getOutputNames(
    Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
  std::vector<std::string> names;
  const size_t count = session.GetOutputCount();
  names.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    auto name = session.GetOutputNameAllocated(i, allocator);
    names.emplace_back(name.get());
  }
  return names;
}

double mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double percentileValue) {
  std::sort(values.begin(), values.end());
  const double index =
      (static_cast<double>(values.size() - 1) * percentileValue) / 100.0;
  return values[static_cast<size_t>(index)];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parseArgs(argsFromArgv(argc, argv));
    if (!std::filesystem::exists(config.modelPath)) {
      std::cerr << "Model file does not exist: " << config.modelPath << '\n';
      return 1;
    }

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "classification-onnx-cpu-benchmark"};
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    std::cout << "Loading ONNX model on CPU: " << config.modelPath << '\n';
    Ort::Session session{env, config.modelPath.c_str(), sessionOptions};

    Ort::AllocatorWithDefaultOptions allocator;
    const std::vector<std::string> inputNames =
        getInputNames(session, allocator);
    const std::vector<std::string> outputNames =
        getOutputNames(session, allocator);

    if (inputNames.empty()) {
      std::cerr << "Model has no inputs\n";
      return 1;
    }
    if (outputNames.empty()) {
      std::cerr << "Model has no outputs\n";
      return 1;
    }

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    if (tensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::cerr << "Expected float input tensor for input: " << inputNames[0]
                << '\n';
      return 1;
    }

    const std::vector<int64_t> inputShape =
        resolveInputShape(tensorInfo.GetShape());
    const size_t inputElements = elementCount(inputShape);
    std::vector<float> inputData(inputElements);
    fillRandomInput(inputData);

    std::cout << "Input: " << inputNames[0] << " [";
    for (size_t i = 0; i < inputShape.size(); ++i) {
      std::cout << inputShape[i] << (i + 1 == inputShape.size() ? "" : ", ");
    }
    std::cout << "] (" << inputElements << " float values)\n";

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(), inputShape.data(),
        inputShape.size());

    const std::vector<const char*> inputNamePtrs = rawNames(inputNames);
    const std::vector<const char*> outputNamePtrs = rawNames(outputNames);

    for (int i = 0; i < config.warmupRuns; ++i) {
      auto outputs =
          session.Run(Ort::RunOptions{nullptr}, inputNamePtrs.data(),
                      &inputTensor, 1, outputNamePtrs.data(),
                      outputNamePtrs.size());
      (void)outputs;
    }

    std::vector<double> timingsMs;
    timingsMs.reserve(static_cast<size_t>(config.benchmarkRuns));
    size_t outputElementCount = 0;
    double outputChecksum = 0.0;

    for (int i = 0; i < config.benchmarkRuns; ++i) {
      const auto start = Clock::now();
      auto outputs =
          session.Run(Ort::RunOptions{nullptr}, inputNamePtrs.data(),
                      &inputTensor, 1, outputNamePtrs.data(),
                      outputNamePtrs.size());
      const auto end = Clock::now();

      timingsMs.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());

      outputElementCount = 0;
      outputChecksum = 0.0;
      for (auto& output : outputs) {
        if (!output.IsTensor()) {
          continue;
        }
        auto outputInfo = output.GetTensorTypeAndShapeInfo();
        outputElementCount += outputInfo.GetElementCount();
        if (outputInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          const auto* values = output.GetTensorData<float>();
          const size_t count = outputInfo.GetElementCount();
          const size_t sampleCount = std::min<size_t>(count, 1024);
          const std::span<const float> outputValues{values, count};
          for (const float value : outputValues.first(sampleCount)) {
            outputChecksum += value;
          }
        }
      }
    }

    std::cout << "CPU benchmark complete\n"
              << "  warmup runs: " << config.warmupRuns << '\n'
              << "  measured runs: " << config.benchmarkRuns << '\n'
              << "  mean: " << mean(timingsMs) << " ms\n"
              << "  p50: " << percentile(timingsMs, P50) << " ms\n"
              << "  p95: " << percentile(timingsMs, P95) << " ms\n"
              << "  min: " << *std::min_element(timingsMs.begin(),
                                                timingsMs.end())
              << " ms\n"
              << "  max: " << *std::max_element(timingsMs.begin(),
                                                timingsMs.end())
              << " ms\n"
              << "  output tensors: " << outputNames.size() << '\n'
              << "  output elements: " << outputElementCount << '\n'
              << "  sampled output checksum: " << outputChecksum << '\n';

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ONNX CPU benchmark failed: " << ex.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "ONNX CPU benchmark failed: unknown exception\n";
    return 1;
  }
}
