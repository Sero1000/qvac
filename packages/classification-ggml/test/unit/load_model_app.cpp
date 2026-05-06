#include <any>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <string>
#include <vector>

#include "model-interface/ClassificationModel.hpp"

template<typename T>
void printFirst(const std::vector<T>& data, size_t count)
{
  std::cout<<"[ ";
  for(size_t i = 0; i < count;++i)
  {
    std::cout<<std::defaultfloat<<(float)data[i]<<", ";
  }
  std::cout<<" ]";
  std::cout.flush();
}

template<typename T>
void printLast(const std::vector<T>& data, size_t count)
{
  std::cout<<"\n[ ";
  const size_t start = data.size() > count ? data.size() - count : 0;
  for (size_t i = start; i < data.size(); ++i) {
    std::cout << std::defaultfloat <<static_cast<float>(data[i]) << ", ";
  }
  // for(size_t i = 0; i < count;++i)
  // {
  //   std::cout<<(float)data[i]<<", ";
  // }
  std::cout<<" ]";
  std::cout.flush();
}

int main() {
  // constexpr const char* kModelPath = "weights/mobilenetv3_3class_v3_fp16.gguf";
  // constexpr const char* kModelPath = "weights/db_mobilenet_v3_large.gguf";
  constexpr const char* kModelPath = "weights/db_mobilenet_v3_large_f32.gguf";
  constexpr const char* kImagePath = "test_image.png";

  try {
    qvac_lib_infer_ggml_classification::ClassificationModel model(kModelPath);
    model.load();
    std::cout << "Model loaded successfully: " << kModelPath << '\n';

    std::ifstream imageFile(kImagePath, std::ios::binary);
    if (!imageFile.is_open()) {
      std::cerr << "Failed to open input image: " << kImagePath << '\n';
      return 1;
    }

    std::vector<uint8_t> imageBytes(
        (std::istreambuf_iterator<char>(imageFile)),
        std::istreambuf_iterator<char>());
    if (imageBytes.empty()) {
      std::cerr << "Input image is empty: " << kImagePath << '\n';
      return 1;
    }

    qvac_lib_infer_ggml_classification::ClassifyInput input;
    input.data = std::move(imageBytes);

    std::any outputAny = model.process(input);
    const auto* output =
        std::any_cast<qvac_lib_infer_ggml_classification::ClassifyOutput>(
            &outputAny);
    if (output == nullptr) {
      std::cerr << "Model returned unexpected output type\n";
      return 1;
    }

    std::cout << "Backbone Inference succeeded for image: " << kImagePath << '\n';
    printFirst(output->data_4, 20);
    printLast(output->data_4, 20);

    // std::cout<<"Output_2 :\n [ ";
    // for(int i = 0; i < 20; ++i)
    // {
    //   std::cout<<output->data_2[i]<<", ";
    // }
    // std::cout<<" ]"<<'\n';
    // std::cout.flush();

    // std::cout<<"Output_3 :\n [ ";
    // for(int i = 0; i < 20; ++i)
    // {
    //   std::cout<<output->data_3[i]<<", ";
    // }
    // std::cout<<" ]"<<'\n';
    // std::cout.flush();

    // std::cout<<"Output_4 :\n [ ";
    // for(int i = 0; i < 20; ++i)
    // {
    //   std::cout<<output->data_4[i]<<", ";
    // }
    // std::cout<<" ]"<<'\n';
    // std::cout.flush();

    // for (const auto& result : output->results) {
    //   std::cout << result.label << ": " << result.confidence << '\n';
    // }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Model load or inference failed: " << ex.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "Model load or inference failed: unknown exception\n";
    return 1;
  }
}

