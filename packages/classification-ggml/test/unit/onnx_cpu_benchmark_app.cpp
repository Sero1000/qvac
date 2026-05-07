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
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t DYNAMIC_BATCH = 1;
constexpr int64_t DYNAMIC_CHANNEL = 3;
constexpr int64_t DYNAMIC_SPATIAL = 1024;
constexpr size_t MAX_ARG_COUNT = 4;
constexpr size_t OUTPUT_SAMPLE_COUNT = 20;
constexpr float SAMPLE_INPUT_VALUE = 0.5F;
constexpr double P50 = 50.0;
constexpr double P95 = 95.0;

struct Config {
  std::string modelPath;
  int warmupRuns{3};
  int benchmarkRuns{20};
};

struct PreparedInputTensor {
  std::vector<float> floatData;
  std::vector<Ort::Float16_t> float16Data;
  Ort::Value tensor{nullptr};
};

struct OutputVectorSample {
  std::vector<double> firstValues;
  std::vector<double> lastValues;
};

void printUsage(std::string_view appName) {
  std::cout << "Usage: " << appName
            << " <model_path> [benchmark_runs] [warmup_runs]\n"
            << "Defaults:\n"
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
  if (args.size() > MAX_ARG_COUNT) {
    throw std::invalid_argument("too many arguments");
  }
  if (args.size() > 1) {
    const std::string& firstArg = args[1];
    if (firstArg == "-h" || firstArg == "--help") {
      printUsage(args[0]);
      std::exit(0);
    }
    config.modelPath = firstArg;
  }
  if (config.modelPath.empty()) {
    printUsage(args[0]);
    throw std::invalid_argument("model_path is required");
  }
  if (args.size() > 2) {
    config.benchmarkRuns = parsePositiveInt(args[2], "benchmark_runs");
  }
  if (args.size() > 3) {
    config.warmupRuns = parsePositiveInt(args[3], "warmup_runs");
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

void fillConstantInput(std::vector<float>& input) {
  std::fill(input.begin(), input.end(), SAMPLE_INPUT_VALUE);
}

void fillConstantInput(std::vector<Ort::Float16_t>& input) {
  for (auto& element : input) {
    element = Ort::Float16_t{SAMPLE_INPUT_VALUE};
  }
}

std::vector<const char*> rawNames(const std::vector<std::string>& names) {
  std::vector<const char*> result;
  result.reserve(names.size());
  for (const auto& name : names) {
    result.push_back(name.c_str());
  }
  return result;
}

PreparedInputTensor createInputTensor(
    ONNXTensorElementDataType inputElementType, size_t inputElements,
    const std::vector<int64_t>& inputShape, const Ort::MemoryInfo& memoryInfo) {
  PreparedInputTensor preparedInput;
  if (inputElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    preparedInput.float16Data.resize(inputElements);
    fillConstantInput(preparedInput.float16Data);
    preparedInput.tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memoryInfo, preparedInput.float16Data.data(),
        preparedInput.float16Data.size(), inputShape.data(), inputShape.size());
    return preparedInput;
  }

  preparedInput.floatData.resize(inputElements);
  fillConstantInput(preparedInput.floatData);
  preparedInput.tensor = Ort::Value::CreateTensor<float>(
      memoryInfo, preparedInput.floatData.data(), preparedInput.floatData.size(),
      inputShape.data(), inputShape.size());
  return preparedInput;
}

OutputVectorSample sampleOutputVector(Ort::Value& output) {
  OutputVectorSample sample;
  auto outputInfo = output.GetTensorTypeAndShapeInfo();
  const size_t count = outputInfo.GetElementCount();
  const size_t firstCount = std::min<size_t>(count, OUTPUT_SAMPLE_COUNT);
  const size_t lastCount = std::min<size_t>(count, OUTPUT_SAMPLE_COUNT);
  sample.firstValues.reserve(firstCount);
  sample.lastValues.reserve(lastCount);

  if (outputInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const auto* values = output.GetTensorData<float>();
    const std::span<const float> outputValues{values, count};
    for (size_t i = 0; i < firstCount; ++i) {
      sample.firstValues.push_back(outputValues[i]);
    }
    for (size_t i = count - lastCount; i < count; ++i) {
      sample.lastValues.push_back(outputValues[i]);
    }
  } else if (outputInfo.GetElementType() ==
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const auto* values = output.GetTensorData<Ort::Float16_t>();
    const std::span<const Ort::Float16_t> outputValues{values, count};
    for (size_t i = 0; i < firstCount; ++i) {
      sample.firstValues.push_back(outputValues[i].ToFloat());
    }
    for (size_t i = count - lastCount; i < count; ++i) {
      sample.lastValues.push_back(outputValues[i].ToFloat());
    }
  }
  return sample;
}

void printOutputValues(std::string_view label,
                       const std::vector<double>& values) {
  std::cout << "  " << label << ":";
  if (values.empty()) {
    std::cout << " <none>\n";
    return;
  }

  for (size_t i = 0; i < values.size(); ++i) {
    std::cout << (i == 0 ? " " : ", ") << values[i];
  }
  std::cout << '\n';
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
    const auto inputElementType = tensorInfo.GetElementType();
    const std::vector<int64_t> rawInputShape = tensorInfo.GetShape();
    if (inputElementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        inputElementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      std::cerr << "Expected float or float16 input tensor for input: "
                << inputNames[0] << '\n';
      return 1;
    }

    const std::vector<int64_t> inputShape =
        resolveInputShape(rawInputShape);
    const size_t inputElements = elementCount(inputShape);

    std::cout << "Input: " << inputNames[0] << " [";
    for (size_t i = 0; i < inputShape.size(); ++i) {
      std::cout << inputShape[i] << (i + 1 == inputShape.size() ? "" : ", ");
    }
    std::cout << "] (" << inputElements
              << (inputElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                      ? " float16 values)\n"
                      : " float values)\n");

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    PreparedInputTensor preparedInput = createInputTensor(
        inputElementType, inputElements, inputShape, memoryInfo);

    const std::vector<const char*> inputNamePtrs = rawNames(inputNames);
    const std::vector<const char*> outputNamePtrs = rawNames(outputNames);

    for (int i = 0; i < config.warmupRuns; ++i) {
      auto outputs =
          session.Run(Ort::RunOptions{nullptr}, inputNamePtrs.data(),
                      &preparedInput.tensor, 1, outputNamePtrs.data(),
                      outputNamePtrs.size());
      (void)outputs;
    }

    std::vector<double> timingsMs;
    timingsMs.reserve(static_cast<size_t>(config.benchmarkRuns));
    size_t outputElementCount = 0;
    OutputVectorSample outputSample;

    for (int i = 0; i < config.benchmarkRuns; ++i) {
      const auto start = Clock::now();
      auto outputs =
          session.Run(Ort::RunOptions{nullptr}, inputNamePtrs.data(),
                      &preparedInput.tensor, 1, outputNamePtrs.data(),
                      outputNamePtrs.size());
      const auto end = Clock::now();

      timingsMs.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());

      outputElementCount = 0;
      outputSample = OutputVectorSample{};
      for (auto& output : outputs) {
        if (!output.IsTensor()) {
          continue;
        }
        auto outputInfo = output.GetTensorTypeAndShapeInfo();
        outputElementCount += outputInfo.GetElementCount();
        if (outputSample.firstValues.empty() &&
            outputSample.lastValues.empty()) {
          outputSample = sampleOutputVector(output);
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
              << "  output elements: " << outputElementCount << '\n';
    printOutputValues("output first 20", outputSample.firstValues);
    printOutputValues("output last 20", outputSample.lastValues);

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ONNX CPU benchmark failed: " << ex.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "ONNX CPU benchmark failed: unknown exception\n";
    return 1;
  }
}
