#ifndef HIAI_ADAPTER_H
#define HIAI_ADAPTER_H

#include <cstdlib>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <algorithm>
#include <iomanip>
#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <unordered_map>
#include <array>
#include <sstream>
#include <android/log.h>
#include <sys/system_properties.h>

#include "hiai_ir_build.h"
#include "HiAiModelManagerService.h"
#include "graph/buffer.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "graph/op/all_ops.h"
#include "graph/compatible/all_ops.h"

#define LOG_TAG "XNN_TEST"
#define ALOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__); \
    printf(__VA_ARGS__)

#define ALOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); \
    printf(__VA_ARGS__)

#define CHECK_SYM_RET(ptr) \
    if(ptr == nullptr || dlerror()) { \
            ALOGE("dlsym failed: %s\n",dlerror()); \
            return FAILED; \
    } \

static const int SUCCESS = 0;
static const int FAILED = -1;



#endif //BUILD_IR_MODEL_TEST_UTIL_H
