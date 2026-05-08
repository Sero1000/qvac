#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "model-interface/ClassificationModel.hpp"
#include "model-interface/ImagePreprocessor.hpp"

#ifndef DEFAULT_DB_MOBILENET_GGUF_PATH
#define DEFAULT_DB_MOBILENET_GGUF_PATH "weights/new_db_mobilenet_v3_large_f16.gguf"
#endif

namespace {

namespace classification = qvac_lib_infer_ggml_classification;

using Clock = std::chrono::steady_clock;

constexpr uint32_t INPUT_WIDTH = 1024;
constexpr uint32_t INPUT_HEIGHT = 1024;
constexpr uint32_t INPUT_CHANNELS = 3;
constexpr std::string_view INPUT_IMAGE_PATH = "../../doctr/tests/test_image.png";
constexpr size_t CHECKSUM_SAMPLE_COUNT = 1024;
constexpr size_t OUTPUT_EDGE_SAMPLE_COUNT = 20;
constexpr size_t MAX_ARG_COUNT = 4;
constexpr size_t DEVICE_ARG_INDEX = 2;
constexpr size_t THREADS_ARG_INDEX = 3;
constexpr double P50 = 50.0;
constexpr double P95 = 95.0;

struct Config {
  std::string modelPath{DEFAULT_DB_MOBILENET_GGUF_PATH};
  int warmupRuns{3};
  int benchmarkRuns{20};
  int threads{0};
  std::string device{"cpu"};
};

void printUsage(std::string_view appName) {
  std::cout << "Usage: " << appName
            << " [model_path] [device] [threads]\n"
            << "Defaults:\n"
            << "  model_path: " << DEFAULT_DB_MOBILENET_GGUF_PATH << '\n'
            << "  benchmark_runs: 20\n"
            << "  warmup_runs: 3\n"
            << "  threads: 0 (ggml default; CPU backend only)\n"
            << "  device: cpu (also accepts gpu, igpu, accelerator, auto, "
               "best, or a registered ggml device name)\n";
}

int parseNonNegativeInt(const std::string& value, std::string_view label) {
  const int parsed = std::stoi(value);
  if (parsed < 0) {
    throw std::invalid_argument(std::string(label) + " must be non-negative");
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
  if (args.size() > DEVICE_ARG_INDEX) {
    config.device = args[DEVICE_ARG_INDEX];
  }
  if (args.size() > THREADS_ARG_INDEX) {
    config.threads = parseNonNegativeInt(args[THREADS_ARG_INDEX], "threads");
  }
  if (args.size() > MAX_ARG_COUNT) {
    throw std::invalid_argument("too many arguments");
  }
  return config;
}

size_t inputElementCount() {
  constexpr size_t maxSize = std::numeric_limits<size_t>::max();
  const size_t width = INPUT_WIDTH;
  const size_t height = INPUT_HEIGHT;
  const size_t channels = INPUT_CHANNELS;
  if (width > maxSize / height || width * height > maxSize / channels) {
    throw std::overflow_error("raw RGB input byte count overflow");
  }
  return width * height * channels;
}

std::vector<float> makeInputTensor() {
  std::vector<float> input(inputElementCount());
  std::fill(input.begin(), input.end(), 1.0F);
  return input;
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

double sampledChecksum(std::span<const float> values) {
  double checksum = 0.0;
  for (const float value :
       values.first(std::min(values.size(), CHECKSUM_SAMPLE_COUNT))) {
    checksum += value;
  }
  return checksum;
}

void printOutputEdgeSamples(std::span<const float> values) {
  const size_t sampleCount = std::min(values.size(), OUTPUT_EDGE_SAMPLE_COUNT);

  std::cout << "  output first " << sampleCount << " elements:";
  for (const float value : values.first(sampleCount)) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';

  std::cout << "  output last " << sampleCount << " elements:";
  for (const float value : values.last(sampleCount)) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';
}

std::vector<uint8_t> loadImageBytes(const std::filesystem::path& imagePath) {
  if (!std::filesystem::exists(imagePath)) {
    throw std::runtime_error("Input image does not exist: " + imagePath.string());
  }

  const uintmax_t fileSize = std::filesystem::file_size(imagePath);
  if (fileSize > static_cast<uintmax_t>(std::numeric_limits<size_t>::max()) ||
      fileSize >
          static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error("input image is too large to read");
  }

  std::vector<char> bytes(static_cast<size_t>(fileSize));
  std::ifstream file(imagePath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open input image: " + imagePath.string());
  }

  file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!file && !bytes.empty()) {
    throw std::runtime_error("Failed to read input image: " + imagePath.string());
  }

  return {bytes.begin(), bytes.end()};
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parseArgs(argsFromArgv(argc, argv));
    if (!std::filesystem::exists(config.modelPath)) {
      std::cerr << "Model file does not exist: " << config.modelPath << '\n';
      return 1;
    }

    std::cout << "Loading GGML model on device '" << config.device
              << "': " << config.modelPath << '\n';
    classification::ClassificationModel model(config.modelPath);
    model.setNumThreads(config.threads);
    model.setDevice(config.device);
    model.load();

    const std::filesystem::path inputImagePath{INPUT_IMAGE_PATH};
    const std::vector<uint8_t> inputImageBytes = loadImageBytes(inputImagePath);
    std::cout << "Input image: " << inputImagePath << " ("
              << inputImageBytes.size() << " bytes)\n";
    std::vector<float> inputTensor = classification::preprocess::preprocessToTensor(
        std::span<const uint8_t>(inputImageBytes.data(), inputImageBytes.size()),
        0, 0, 0);
    std::cout << "Preprocessed input elements: " << inputTensor.size() << '\n';
    std::cout << "Threads: "
              << (config.threads == 0 ? "ggml default"
                                      : std::to_string(config.threads))
              << '\n';
    std::cout << "Device: " << config.device << '\n';

    for (int i = 0; i < config.warmupRuns; ++i) {
      const auto output = model.runTensor(inputTensor);

      (void)output;
    }

    std::vector<double> timingsMs;
    timingsMs.reserve(static_cast<size_t>(config.benchmarkRuns));
    size_t outputElementCount = 0;
    double outputChecksum = 0.0;
    std::vector<float> lastOutput;

    for (int i = 0; i < config.benchmarkRuns; ++i) {
      const auto start = Clock::now();
      const auto output = model.runTensor(inputTensor);
      const auto end = Clock::now();

      timingsMs.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());

      outputElementCount = output.data_4.size();
      outputChecksum = sampledChecksum(output.data_4);
      lastOutput = output.data_4;
    }

    std::cout << "GGML benchmark complete\n"
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
              << "  output tensors: 1\n"
              << "  output elements: " << outputElementCount << '\n'
              << "  sampled output checksum: " << outputChecksum << '\n';


    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "GGML CPU benchmark failed: " << ex.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "GGML CPU benchmark failed: unknown exception\n";
    return 1;
  }
}
