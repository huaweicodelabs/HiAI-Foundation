  
### 零拷贝新增2个接口
  * 新增接口1：Init
  * 功能：支持创建带fd信息的输入和输出AiTensor，是通用的AiTensor初始化接口。新增接口2是基于此接口，增加了buffer_handle_t结构体解析过程。
  * 函数定义头文件：ddk/ai_ddk_lib/include/HiAiModelManagerType.h
  * 函数实现源文件：bin/ap/hardware/hiai_ddk/hiai_api/jni/AiTensor.cpp
  * 原型如下：
```
struct NativeHandle {
    int fd;
    int size;
    int offset;
};
AIStatus Init(const NativeHandle& handle, // NativeHandle是hiai自定义的带fd信息的数据结构
              const TensorDimension* dim, // 使用Client的GetModelIOTensorDim接口获取到的dim信息
              HIAI_DataType pdataType
);
)

```
  * 新增接口2：HIAI_CreateAiPPTensorFromHandle
  * 功能：创建支持ION buffer_handle_t作为输入的含AIPP的输入tensor，自动解析buffer handle的stride信息，并填充到Crop参数，用于去除由GPU处理自动补齐的padding数据
  * 函数定义头文件：ddk/ai_ddk_lib/include/HiAiAippPara.h:265
  * 函数实现源文件：bin/ap/hardware/hiai_ddk/hiai_api/jni/HiAiAippPara.cpp
  * 原型如下：
```
std::shared_ptr<AippTensor> HIAI_CreateAiPPTensorFromHandle(
        buffer_handle_t& handle,   // 新增参数：输入的ION buffer handle
        const TensorDimension* dim, // 使用Client的GetModelIOTensorDim接口获取到的输入dim信息
        AiTensorImage_Format imageFormat = AiTensorImage_INVALID  // 输入图像的格式
);
```

### 示例：主要介绍了新增接口2的使用方法，Status Test(const TestCase& test)函数是用例主体
  * (1). 模拟用户已经获取到了buffer_handle_t
```
    // 模拟用户直接拿到buffer_handle_t
    buffer_handle_t handle = nullptr;
    get_ion_handle::InitBufferHandle(handle, test.inputH, test.inputW);
    CHECK_NULL(handle);

    // 往ION内存中写测试数据, 实际用户拿到的是有图像数据的。
    auto* ionMemAddr = (uint8_t*)mmap(NULL, handle->data[10], PROT_READ | PROT_WRITE, MAP_SHARED, handle->data[0], 0);
    for (int i = 0; i < handle->data[10]; ++i) {
        ionMemAddr[i] = i % 256;
    }
```
  * (2). 创建AippDtcPara参数，用于对输入图像数据进行归一化使用，参数已经配置好，单通道，从0~255归一化到0~1，均值0（不减均值）
```
// TODO ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 创建带ION的AIPP参数，并自动设置Crop和DTC（归一化到0~1） ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
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
    hiai::AippDtcPara para;
    GetNormalizeAippDtcPara(para);
```

  * (3). CreateModelBuff、BuildIRMode、client->Load过程省略，与之前流程无差异。需要注意的时，构建IR图的时候，因为使用了AIPP，因此输入节点需要从原来的op::Data节点替换成op::DynamicImageData，示例如下：
```
#define ALIGN_CEIL(N, n) (((N) + (n) -1 ) / (n) * (n))

    auto data = hiai::op::DynamicImageData("Placeholder");
    TensorDesc inputDesc(Shape({1, 864, 480, 1}), FORMAT_NHWC, DT_UINT8);
    data.update_input_desc_x(inputDesc);
    int alignH = ALIGN_CEIL(inputDesc.GetShape().GetDim(1), 16); // 此对齐不固定，根据实际情况计算。
    int alignW = ALIGN_CEIL(inputDesc.GetShape().GetDim(2), 64); // 此对齐不固定，根据实际情况计算。
    // 设置原始输入图像的内存占用大小
    data.set_attr_max_src_image_size(3 * alignH * alignW * 2);
    data.set_attr_image_type("JPEG");
```

  * (4). client->GetModelIOTensorDim之后，使用buffer_handle_t作为ion输入创建AippTensor（输出未设置ion，使用方法比较简单，可自行设置）
```
    // 创建带ION buffer输入的AippTensor，自动解析buffer_handle_t中的stride信息，并设置Crop参数
    std::shared_ptr<hiai::AippTensor> inputTensor = HIAI_CreateAiPPTensorFromHandle(inputHandle, &inputDims[i], imageFormat);
    CHECK_EXE(inputTensor, return nullptr);
```

  * (5) 为AippTensor设置AIPP DTC参数
```
    // 手动为aippTensor添加DTC（数据转换及归一化）参数
    inputTensor->GetAippParas().at(0)->SetDtcPara(dtcPara);
```

  * (6) RunModel及之后流程无差异
