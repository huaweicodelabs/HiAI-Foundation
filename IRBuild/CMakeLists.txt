cmake_minimum_required(VERSION 3.17)
project(build_ir_model)

set(CMAKE_CXX_STANDARD 11)

include_directories(
        ${PROJECT_SOURCE_DIR}/jni
        ${PROJECT_SOURCE_DIR}/ddk/ai_ddk_lib/include
        ${PROJECT_SOURCE_DIR}/ddk/ai_ddk_lib/include/graph
        $ENV{NDK_PATH}/sysroot/usr/include
)

file(GLOB SRC0 "${PROJECT_SOURCE_DIR}/jni/*.cpp")
file(GLOB SRC1 "${PROJECT_SOURCE_DIR}/jni/*/*.cpp")

add_library(
        other_cpp
        SHARED
        ${SRC0}
        ${SRC1}
        jni/test_contains_npu.cpp jni/test_single_op.cpp jni/test_util.h jni/check.h)