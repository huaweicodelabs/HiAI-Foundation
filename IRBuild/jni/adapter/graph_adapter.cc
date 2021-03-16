// 此处包含C++头文件
#include<iostream>
#include"adapter.h"
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

void* CreateGraph(string name) {
    auto ptr = new ge::Graph("name");
    return (void*)ptr;
}

void SetGraph(void* graphPtr,std::vector<void*> inputsPtr, std::vector<void*> outputPtr){
    std::vector<ge::Operator> inputs;
    std::vector<ge::Operator> outputs;
    for(auto opPtr : inputsPtr) {
        inputs.push_back(*(ge::Operator*)opPtr);
    }
    for(auto opPtr : outputPtr) {
        outputs.push_back(*(ge::Operator*)opPtr);
    }
    ge::Graph* graph = (ge::Graph*)graphPtr;
    (*graph).SetInputs(inputs).SetOutputs(outputs);
}

void DestroyGraph(void* ptr) {
    ge::Graph* obj = (ge::Graph*)ptr;
    delete obj;
}

void PrintTensorInfo(const string& msg, const std::shared_ptr<hiai::AiTensor>& tensor) {
    auto dims = tensor->GetTensorDimension();
    std::cout << msg << " NCHW: " << dims.GetNumber() << ", " << dims.GetChannel() << ", " << dims.GetHeight() << ", "
              << dims.GetWidth() << std::endl;
}

bool WriteFile(const void* data, size_t size, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        ALOGE("%s open failed.\n", path.c_str());
        return false;
    }
    file.write((const char*)data, size);
    file.flush();
    file.close();
    return true;
}

void* CreateModelClient(const std::string modelName, void* graph,
                 std::vector<std::shared_ptr<hiai::AiTensor>>* inputTensors,
                 std::vector<std::shared_ptr<hiai::AiTensor>>* outputTensors) {
    
    ge::Graph* irGraph = (ge::Graph*)graph;
    ge::Model irModel("model", modelName);
    irModel.SetGraph(*irGraph);
    domi::HiaiIrBuild irBuild;
    domi::ModelBufferData omModelBuf;
    ge::Buffer buffer;
    irModel.Save(buffer);
    WriteFile(buffer.GetData(), buffer.GetSize(),  
              "/data/local/tmp/" + modelName + ".irpb");
    if (!irBuild.CreateModelBuff(irModel, omModelBuf)) {
        ALOGE("ERROR: build alloc om failed.\n");
        return nullptr;
    }
    if (!irBuild.BuildIRModel(irModel, omModelBuf)) {
        irBuild.ReleaseModelBuff(omModelBuf);
        ALOGE("ERROR: build ir model failed.\n");
        return nullptr;
    }
    auto client = new hiai::AiModelMngerClient();
    int retCode = client->Init(nullptr);
    if (retCode != hiai::AI_SUCCESS) {
        ALOGE("ERROR: build init hiai::AiModelManagerClient failed(retCode=%d)\n", retCode);
        return nullptr;
    }
    auto modelDesc = std::make_shared<hiai::AiModelDescription>(modelName, 3, 0, 0, 0);
    modelDesc->SetModelBuffer(omModelBuf.data, omModelBuf.length);
    std::vector<std::shared_ptr<hiai::AiModelDescription>> modelDescs;
    modelDescs.push_back(modelDesc);
    retCode = client->Load(modelDescs);
    if (retCode != 0) {
        ALOGE("ERROR: hiai::AiModelMngerClient load model failed.\n");
        return nullptr;
    }
    std::vector<hiai::TensorDimension> inputDims;
    std::vector<hiai::TensorDimension> outputDims;
    retCode = client->GetModelIOTensorDim(modelName, inputDims, outputDims);
    if (retCode != 0) {
        ALOGE("ERROR: get IO tensor failed retCode=%d.\n", retCode);
        return nullptr;
    }
    inputTensors->clear();
    outputTensors->clear();
    for (int i = 0; i < inputDims.size(); i++) {
        std::shared_ptr<hiai::AiTensor> inputTensor = std::make_shared<hiai::AiTensor>();
        inputTensor->Init(&inputDims[i]);
        inputTensors->push_back(inputTensor);
        PrintTensorInfo("input_tensor_" + std::to_string(i), inputTensor);
    }
    for (int i = 0; i < outputDims.size(); i++) {
        std::shared_ptr<hiai::AiTensor> outputTensor = std::make_shared<hiai::AiTensor>();
        outputTensor->Init(&outputDims[i]);
        outputTensors->push_back(outputTensor);
        PrintTensorInfo("output_tensor_" + std::to_string(i), outputTensor);
    }
    if (client != nullptr) {
        if (!WriteFile(omModelBuf.data, omModelBuf.length, 
                       "/data/local/tmp/" + modelName)) {
            ALOGE("ERROR: save om model failed.\n");
        }
    }
    irBuild.ReleaseModelBuff(omModelBuf);
    return client;
}

void DestroyModelClient(void* client) {
    hiai::AiModelMngerClient* obj = (hiai::AiModelMngerClient*)client;
    delete obj;
}

bool RunModelClient(void* clientPtr,
              std::string& modelName,
              std::vector<std::shared_ptr<hiai::AiTensor>>* inputTensors,
              std::vector<std::shared_ptr<hiai::AiTensor>>* outputTensors) {
    hiai::AiContext context;
    string key = "model_name";
    const string& value = modelName;
    context.AddPara(key, value);
    int istamp;

    hiai::AiModelMngerClient* client = (hiai::AiModelMngerClient*)clientPtr;
    int retCode = client->Process(context, *inputTensors, *outputTensors, 1000, istamp);
    if (retCode) {
        ALOGE("Run model failed. retCode=%d\n", retCode);
        return false;
    }
    ALOGI("inferenc success!\n ");
    return true;
}

#ifdef __cplusplus
}
#endif