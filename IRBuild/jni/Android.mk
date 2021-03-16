LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := test
LOCAL_SRC_FILES := test_single_op.cpp \

LOCAL_C_INCLUDES := $(LOCAL_PATH)/../ddk/ai_ddk_lib/include \
					$(LOCAL_PATH)/../dkk/ai_ddk_lib/include/graph \
          $(LOCAL_PATH)/adapter \
					$(LOCAL_PATH)/ \

DDK_LIBRARY_PATH := $(LOCAL_PATH)/../ddk/ai_ddk_lib

APP_ALLOW_MISSING_DEPS := true
LOCAL_ALLOW_UNDEFINED_SYMBOLS := true
LOCAL_FORCE_STATIC_EXECUTABLE := true
LOCAL_LDLIBS += \
  -llog \
  -lc \
  -lm \
  -ldl \
  -landroid \

LOCAL_SHARED_LIBRARIES += \
#  libhiai_ir \
#  libhiai_ir_build \
#  libhiai \

CPPFLAGS=-stdlib=libstdc++
LDLIBS=-lstdc++
LOCAL_CFLAGS += -std=c++14 -frtti

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := XNN_NPU
LOCAL_SRC_FILES := adapter/op_adapter.cc \
                   adapter/tensor_adapter.cc \
                   adapter/graph_adapter.cc

LOCAL_C_INCLUDES := $(LOCAL_PATH)/adapter \
                    $(LOCAL_PATH)/../ddk/ai_ddk_lib/include \
					          $(LOCAL_PATH)/../dkk/ai_ddk_lib/include/graph \

LOCAL_ALLOW_UNDEFINED_SYMBOLS := true
APP_ALLOW_MISSING_DEPS := true
LOCAL_CFLAGS += -std=c++14
CPPFLAGS=-stdlib=libstdc++

LOCAL_LDLIBS += \
  -llog \
  -lc \
  -lm \
  -ldl \
  -landroid \

LOCAL_SHARED_LIBRARIES += \
  libhiai_ir \
  libhiai_ir_build \
  libhiai \

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libhiai_ir
LOCAL_SRC_FILES := $(DDK_LIBRARY_PATH)/lib64/libhiai_ir.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libhiai_ir_build
LOCAL_SRC_FILES := $(DDK_LIBRARY_PATH)/lib64/libhiai_ir_build.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libhiai
LOCAL_SRC_FILES := $(DDK_LIBRARY_PATH)/lib64/libhiai.so
include $(PREBUILT_SHARED_LIBRARY)
