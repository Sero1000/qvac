cmake_minimum_required(VERSION 3.25)

project(ggml_classification_tests LANGUAGES C CXX)

find_package(GTest CONFIG REQUIRED)
find_package(onnxruntime CONFIG QUIET)

option(ENABLE_COVERAGE "Enable coverage instrumentation for unit tests" OFF)

set(TEST_SOURCES
  classification_model_test.cpp
  preprocessor_test.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/ClassificationModel.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/ImagePreprocessor.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/MobileNetGraph.cpp
)

add_executable(addon-test ${TEST_SOURCES})
target_compile_features(addon-test PRIVATE cxx_std_20)

target_compile_options(addon-test PRIVATE -Wall -Wextra -Wfatal-errors -g)
target_compile_definitions(addon-test PRIVATE GTEST_HAS_POSIX_RE=0)

if(NOT WIN32)
  target_compile_options(addon-test PRIVATE -fno-omit-frame-pointer)
endif()

target_include_directories(
  addon-test
  PRIVATE
  ${CMAKE_SOURCE_DIR}/addon/src
  ${QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS}
  ${STB_INCLUDE_DIRS}
)

target_link_libraries(
  addon-test
  PRIVATE
    ggml::ggml
    ggml::ggml-base
    ggml::ggml-cpu
    GTest::gtest
    GTest::gtest_main
)

set(LOAD_APP_SOURCES
  load_model_app.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/ClassificationModel.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/ImagePreprocessor.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/MobileNetGraph.cpp
)

add_executable(classification-load-model-app ${LOAD_APP_SOURCES})
target_compile_features(classification-load-model-app PRIVATE cxx_std_20)
target_compile_options(classification-load-model-app PRIVATE -Wall -Wextra -Wfatal-errors -g)

if(NOT WIN32)
  target_compile_options(classification-load-model-app PRIVATE -fno-omit-frame-pointer)
endif()

target_include_directories(
  classification-load-model-app
  PRIVATE
  ${CMAKE_SOURCE_DIR}/addon/src
  ${QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS}
  ${STB_INCLUDE_DIRS}
)

target_link_libraries(
  classification-load-model-app
  PRIVATE
    ggml::ggml
    ggml::ggml-base
    ggml::ggml-cpu
)

if(NOT WIN32)
  target_link_libraries(classification-load-model-app PRIVATE -fsanitize=address)
endif()

if(WIN32)
  target_link_libraries(classification-load-model-app PRIVATE msvcrt.lib)
endif()

if(onnxruntime_FOUND)
  add_executable(classification-onnx-cpu-benchmark-app
    onnx_cpu_benchmark_app.cpp
  )
  target_compile_features(classification-onnx-cpu-benchmark-app PRIVATE cxx_std_20)
  target_compile_options(classification-onnx-cpu-benchmark-app PRIVATE -Wall -Wextra -Wfatal-errors -g)
  target_compile_definitions(
    classification-onnx-cpu-benchmark-app
    PRIVATE
      DEFAULT_DB_MOBILENET_ONNX_PATH="${CMAKE_SOURCE_DIR}/../../onnx_models/db_mobilenet_v3_large.onnx"
  )
  target_link_libraries(
    classification-onnx-cpu-benchmark-app
    PRIVATE
      onnxruntime::onnxruntime
  )
  if(WIN32)
    target_link_libraries(classification-onnx-cpu-benchmark-app PRIVATE msvcrt.lib)
  endif()
else()
  message(STATUS "onnxruntime not found; skipping classification ONNX CPU benchmark app")
endif()

if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(addon-test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(addon-test PRIVATE -fprofile-instr-generate)
  endif()
endif()

if(NOT WIN32)
  target_link_libraries(addon-test PRIVATE -fsanitize=address)
endif()

if(WIN32)
  target_link_libraries(addon-test PRIVATE msvcrt.lib)
endif()

add_test(NAME ClassificationModelTests COMMAND addon-test)
set_tests_properties(ClassificationModelTests PROPERTIES
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  TIMEOUT 600)
