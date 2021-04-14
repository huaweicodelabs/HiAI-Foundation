#include "test_util.h"
#include "EGL/egl.h"
#include "EGL/eglext.h"
#include <dlfcn.h>
#include <android/hardware_buffer.h>
#include <asm-generic/mman-common.h>
#include <sys/mman.h>

using namespace ge;
using namespace std;
using namespace hiai;
using namespace test_case;
using namespace test_util;
using namespace ir_model;
#define EGL_EGLEXT_PROTOTYPES

namespace testcase {

bool CheckResult()
{
    return true;
}

namespace get_ion_handle {
using GetHardwareBufferNativeHandlePtr = const native_handle_t* (*)(const AHardwareBuffer* buffer);

GetHardwareBufferNativeHandlePtr g_getNativeHandle = nullptr;

AHardwareBuffer* CreateHandwareBufferObj(uint32_t height, uint32_t width)
{
    AHardwareBuffer_Desc usage;
    usage.format = AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420;
    usage.height = height;
    usage.width = width;
    usage.layers = 1;
    usage.rfu0 = 0;
    usage.rfu1 = 0;
    usage.stride = 10;

    usage.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                  AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                  AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;
    AHardwareBuffer* graphicBuf;
    if (AHardwareBuffer_allocate(&usage, &graphicBuf)) {
        ALOGE("AHardwareBuffer_allocate hardwarebuffer failed.\n");
        return nullptr;
    }
    ALOGI("AHardwareBuffer_allocate hardwarebuffer success.\n");

    AHardwareBuffer_Desc usage1;
    AHardwareBuffer_describe(graphicBuf, &usage1);
    ALOGI("format:%d\n", usage1.format);
    ALOGI("height:%d\n", usage1.height);
    ALOGI("width :%d\n", usage1.width);
    ALOGI("layers:%d\n", usage1.layers);
    ALOGI("rfu0  :%d\n", usage1.rfu0);
    ALOGI("rfu1  :%lu\n", usage1.rfu1);
    ALOGI("stride:%d\n", usage1.stride);  // 512
    ALOGI("usage :%lu\n", usage1.usage);
    return graphicBuf;
}

bool InitNativeWindowLib()
{
    void* libnativewindow = dlopen("libnativewindow.so", RTLD_LAZY);
    if (libnativewindow == nullptr) {
        ALOGE("dloepn libnativewindow.so failed.\n");
        return false;
    }
    ALOGI("dloepn libnativewindow.so success.\n");

    void* funcPtr = dlsym(libnativewindow, "AHardwareBuffer_getNativeHandle");

    if (funcPtr == nullptr) {
        ALOGE("dlsym AHardwareBuffer_getNativeHandle failed.\n");
        return false;
    }
    g_getNativeHandle = reinterpret_cast<GetHardwareBufferNativeHandlePtr>(funcPtr);
    ALOGI("AHardwareBuffer_getNativeHandle function pointer address: %p\n", g_getNativeHandle);
    return true;
}

const native_handle_t* GetNativeHandle(const AHardwareBuffer* graphicBuf)
{
    const native_handle_t* handle = g_getNativeHandle(graphicBuf);
    if (handle == nullptr) {
        ALOGE("AHardwareBuffer_getNativeHandleP return failed\n");
        return nullptr;
    }
    ALOGI("AHardwareBuffer_getNativeHandleP return handle pointer address: %p\n", handle);
    return handle;
}

void ShowNativeHandle(const native_handle_t* handle)
{
    ALOGI("handle->data[0]: %d\n", handle->data[0]);
    ALOGI("handle->numFds: %d\n", handle->numFds);
    ALOGI("handle->numInts: %d\n", handle->numInts);
    for (int i = 0; i < handle->numInts + handle->numFds; i++) {
        ALOGI("handle->data[%d]: %d\n", i, handle->data[i]);
    }
}

bool InitBufferHandle(buffer_handle_t& bufferHandle, uint32_t testH, uint32_t testW)
{
    if (!InitNativeWindowLib()) {
        return false;
    }
    AHardwareBuffer* graphicBuf = CreateHandwareBufferObj(testH, testW);
    if (graphicBuf == nullptr) {
        return false;
    }
    auto handle = GetNativeHandle(graphicBuf);
    if (handle == nullptr) {
        return false;
    }
    ShowNativeHandle(handle);
    bufferHandle = static_cast<buffer_handle_t>(handle);
    return true;
}
}

namespace build_graph {
void SetConvolution(std::string& name, hiai::op::Convolution conv,
    hiai::op::Const& filter, const hiai::Shape& filterShape, hiai::op::Const& bias, const hiai::Shape& biasShape)
{
    // op -- filter
    hiai::TensorDesc convWeightDesc(filterShape, FORMAT_NCHW, DT_FLOAT);
    vector<float> convWeightValue(Prod(filterShape.GetDims()), 1);
    SetConstData(filter, convWeightDesc, (uint8_t*)convWeightValue.data(), convWeightValue.size() * sizeof(float));

    // op -- bias
    hiai::TensorDesc convBiasDesc(biasShape, FORMAT_NCHW, DT_FLOAT);
    vector<float> convBiasValue(Prod(biasShape.GetDims()), 1);
    SetConstData(bias, convBiasDesc, (uint8_t*)convBiasValue.data(), convBiasValue.size() * sizeof(float));

    // op
    conv.set_input_filter(filter)
        .set_input_bias(bias)
        .set_attr_dilations({1, 1})
        .set_attr_strides({1, 1})
        .set_attr_pad_mode("SAME")
        .set_attr_pads({0, 0, 0, 0});
}

void SetDeconv(std::string& name, ge::op::Deconvolution deconv,
    hiai::op::Const& filter, const hiai::Shape& filterShape, hiai::op::Const& bias, const hiai::Shape& biasShape)
{
    // op -- filter
    hiai::TensorDesc convWeightDesc(filterShape, FORMAT_NCHW, DT_FLOAT);
    vector<float> convWeightValue(Prod(filterShape.GetDims()), 1);
    SetConstData(filter, convWeightDesc, (uint8_t*)convWeightValue.data(), convWeightValue.size() * sizeof(float));

    // op -- bias
    hiai::TensorDesc convBiasDesc(biasShape, FORMAT_NCHW, DT_FLOAT);
    vector<float> convBiasValue(Prod(biasShape.GetDims()), 1);
    SetConstData(bias, convBiasDesc, (uint8_t*)convBiasValue.data(), convBiasValue.size() * sizeof(float));

    // //input size
    // TensorDesc input_desc(Shape({4}), FORMAT_NCHW, DT_INT32);
    // ge::AttrValue::TENSOR input_tensor = std::make_shared<ge::Tensor>(input_desc);
    // vector<int32_t> dataValuec = {1, 8, 864, 480};
    // input_tensor->SetData((uint8_t*)dataValuec.data(), 4 * sizeof(int32_t));
    // ge::op::Const input_op = ge::op::Const(name + "_input").set_attr_value(input_tensor);

    // op
    deconv.set_input_filter(filter)
          .set_input_bias(bias)
          .set_attr_dilation({1, 1})
          .set_attr_stride({2, 2})
          .set_attr_group(1)
          .set_attr_pad_mode(6)
          .set_attr_bias_term(1)
          .set_attr_kernel({4, 4})
          .set_attr_num_output(1)
          .set_attr_pad({0, 0, 0, 0});
}

void SetConvTranspose(std::string& name, hiai::op::ConvTranspose deconv,
    hiai::op::Const& filter, const hiai::Shape& filterShape, hiai::op::Const& bias, const hiai::Shape& biasShape)
{
    // op -- filter
    hiai::TensorDesc convWeightDesc(filterShape, FORMAT_NCHW, DT_FLOAT);
    vector<float> convWeightValue(Prod(filterShape.GetDims()), 1);
    SetConstData(filter, convWeightDesc, (uint8_t*)convWeightValue.data(), convWeightValue.size() * sizeof(float));

    // op -- bias
    hiai::TensorDesc convBiasDesc(biasShape, FORMAT_NCHW, DT_FLOAT);
    vector<float> convBiasValue(Prod(biasShape.GetDims()), 1);
    SetConstData(bias, convBiasDesc, (uint8_t*)convBiasValue.data(), convBiasValue.size() * sizeof(float));

    // op
    deconv.set_input_filter(filter)
          .set_input_bias(bias)
          .set_attr_dilations({1, 1})
          .set_attr_strides({2, 2})
          .set_attr_groups(1)
          .set_attr_pad_mode("SAME")
          .set_attr_pads({0, 0, 0, 0});
}

bool BuildIonTestGraph(ge::Graph& graph)
{
    // input op
    auto data = hiai::op::DynamicImageData("Placeholder");
    TensorDesc inputDesc(Shape({1, 8, 16, 1}), FORMAT_NHWC, DT_UINT8);
    data.update_input_desc_x(inputDesc);
    int alignH = ALIGN_CEIL(inputDesc.GetShape().GetDim(1), 16);
    int alignW = ALIGN_CEIL(inputDesc.GetShape().GetDim(2), 64);
    data.set_attr_max_src_image_size(3 * alignH * alignW * 2);
    data.set_attr_image_type("JPEG");

    auto output = ge::op::Activation("Activation")
        .set_input_x(data)
        .set_attr_mode(1);

    std::vector<Operator> inputs{data};
    std::vector<Operator> outputs{output};
    graph.SetInputs(inputs).SetOutputs(outputs);
    return true;
}

bool BuildGraph(ge::Graph& graph)
{
    // input op
    auto data = hiai::op::DynamicImageData("Placeholder");
    TensorDesc inputDesc(Shape({1, 864, 480, 1}), FORMAT_NHWC, DT_UINT8);
    data.update_input_desc_x(inputDesc);
    int alignH = ALIGN_CEIL(inputDesc.GetShape().GetDim(1), 16);
    int alignW = ALIGN_CEIL(inputDesc.GetShape().GetDim(2), 64);
    data.set_attr_max_src_image_size(3 * alignH * alignW * 2);
    data.set_attr_image_type("JPEG");

    string convName = "Conv_Conv2D";
    auto convOp = hiai::op::Convolution(convName).set_input_x(data);
    auto convFilter = hiai::op::Const(convName + "_filter");
    auto convBias = hiai::op::Const(convName + "_bias");
    {
        hiai::Shape filterShape = Shape({8, 1, 3, 3});
        hiai::Shape biasShape = Shape({8});
        SetConvolution(convName, convOp, convFilter, filterShape, convBias, biasShape);
    }
    auto convReluOp = ge::op::Activation(convName + "_relu")
        .set_input_x(convOp)
        .set_attr_mode(1);

    string conv1Name = "Conv1_Conv2D";
    auto conv1Op = hiai::op::Convolution(conv1Name).set_input_x(convReluOp);
    auto conv1Filter = hiai::op::Const(conv1Name + "_filter");
    auto conv1Bias = hiai::op::Const(conv1Name + "_bias");
    {
        hiai::Shape filter1Shape = Shape({8, 8, 3, 3});
        hiai::Shape bias1Shape = Shape({8});
        SetConvolution(conv1Name, conv1Op, conv1Filter, filter1Shape, conv1Bias, bias1Shape);
    }
    auto conv1ReluOp = ge::op::Activation(conv1Name + "_relu")
        .set_input_x(conv1Op)
        .set_attr_mode(1);

    string conv2Name = "Conv2_Conv2D";
    auto conv2Op = hiai::op::Convolution(conv2Name).set_input_x(conv1ReluOp);
    auto conv2Filter = hiai::op::Const(conv2Name + "_filter");
    auto conv2Bias = hiai::op::Const(conv2Name + "_bias");
    {
        hiai::Shape filter2Shape = Shape({8, 8, 3, 3});
        hiai::Shape bias2Shape = Shape({8});
        SetConvolution(conv2Name, conv2Op, conv2Filter, filter2Shape, conv2Bias, bias2Shape);
    }
    auto conv2ReluOp = ge::op::Activation(conv2Name + "_relu")
        .set_input_x(conv2Op)
        .set_attr_mode(1);

    string deconvName = "Conv2d_transpose_conv2d_transpose_1";
    // auto deconvOp = ge::op::Deconvolution(deconvName).set_input_x(conv2ReluOp);
    auto deconvOp = hiai::op::ConvTranspose(deconvName).set_input_x(conv2ReluOp);
    auto deconvFilter = hiai::op::Const(deconvName + "_filter");
    auto outputShape = hiai::op::Const(deconvName + "_output");
    auto deconvBias = hiai::op::Const(deconvName + "_bias");
    {
        hiai::TensorDesc outDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
        vector<int32_t> outShapeValue = {
            (int32_t)inputDesc.GetShape().GetDim(0),
            (int32_t)inputDesc.GetShape().GetDim(3),
            (int32_t)inputDesc.GetShape().GetDim(1) * 2,
            (int32_t)inputDesc.GetShape().GetDim(2) * 2,
        };
        SetConstData(outputShape, outDesc, (uint8_t*)outShapeValue.data(), outShapeValue.size() * sizeof(int32_t));

        hiai::Shape deconvFilterShape = Shape({8, 1, 4, 4});
        hiai::Shape deconvBiasShape = Shape({1});
        // SetDeconv(deconvName, deconvOp, deconvFilter, deconvFilterShape, deconvBias, deconvBiasShape);
        SetConvTranspose(deconvName, deconvOp, deconvFilter, deconvFilterShape, deconvBias, deconvBiasShape);

        deconvOp.set_input_output_shape(outputShape);
    }

    std::vector<Operator> inputs{data};
    std::vector<Operator> outputs{deconvOp};
    graph.SetInputs(inputs).SetOutputs(outputs);
    return true;
}
}

void GetNormalizeAippDtcPara(hiai::AippDtcPara& para)
{
    para.pixelMeanChn0 = 0;
    para.pixelMeanChn1 = 0;
    para.pixelMeanChn2 = 0;
    para.pixelMeanChn3 = 0;
    para.pixelMinChn0 = 0;
    para.pixelMinChn1 = 0;
    para.pixelMinChn2 = 0;
    para.pixelMinChn3 = 0;
    para.pixelVarReciChn0 = 1.0 / 255;
    para.pixelVarReciChn1 = 1.0 / 255;
    para.pixelVarReciChn2 = 1.0 / 255;
    para.pixelVarReciChn3 = 1.0 / 255;
}

bool Test(const TestCase& test)
{
    // 模拟用户直接拿到buffer_handle_t
    buffer_handle_t handle = nullptr;
    get_ion_handle::InitBufferHandle(handle, test.inputH, test.inputW);
    if (handle == nullptr) {
        return false;
    }

    // 往ION内存中写测试数据, 实际用户拿到的是有图像数据的。
    auto* ionMemAddr = (uint8_t*)mmap(NULL, handle->data[10], PROT_READ | PROT_WRITE, MAP_SHARED, handle->data[0], 0);
    for (int i = 0; i < handle->data[10]; ++i) {
        ionMemAddr[i] = i % 256;
    }

    std::string modelName = "./output/" + test.caseName + ".om";
    Graph irGraph = Graph("irGraph");
    cout << "============== CaseName: " << test.caseName << endl << endl;
    test.func(irGraph);
    std::vector<std::shared_ptr<hiai::AiTensor>> itensors;
    std::vector<std::shared_ptr<hiai::AiTensor>> otensors;
    ge::Model irModel("model", modelName);
    irModel.SetGraph(irGraph);

    // TODO ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 创建带ION的AIPP参数，并自动设置Crop和DTC（归一化到0~1） ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    hiai::AippDtcPara para;
    GetNormalizeAippDtcPara(para);
    // handle是buffer_handle_t类型，已经确认，ByteNN可以拿到这个数据结构，当前作为输入用，输出也可以设置为ION buffer
    auto client = BuildWithION(modelName, irModel, &itensors, &otensors, handle, AiTensorImage_YUV420SP_U8, para);
    if (client == nullptr) {
        return false;
    }
    // TODO ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 创建带ION的AIPP参数，并自动设置Crop和DTC（归一化到0~1） ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


    // 与普通流程一致
    for (auto& itensor : itensors) {
        PrintTensorData<uint8_t>(itensor, 0, 32);
        SaveTensorData<uint8_t>(itensor, "/data/local/tmp/output/input_u8.bin");
    }

    // build and run model
    if (!RunModel(client, modelName, &itensors, &otensors)) {
        ALOGE("Run %s failed.\n", modelName.c_str());
        return false;
    }
    // fetch and print output tensors
    for (const auto& tensor: otensors) {
        PrintTensorData<float>(tensor, 0, 32);
        SaveTensorData<uint8_t>(tensor, "/data/local/tmp/output/output_f32.bin");
    }
    cout << "-------------  " << test.caseName << "  ---  " << CheckResult() << endl;
    return SUCCESS;
}
}
using namespace testcase;

int main(int argc, char* argv[])
{
    TestCase caseList[] = {
        {"ion_test", build_graph::BuildIonTestGraph, false, 8,   16},
        {"ir_graph",   build_graph::BuildGraph,  false, 864, 480},
    };
    for (const TestCase& tc : caseList) {
        Test(tc);
    }
    std::cout << "all done." << std::endl;
    return 0;
}

