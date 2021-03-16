#include "adapter.h"
#include <dlfcn.h>

using namespace std;

namespace test_case {

void* (*CreateTensorDesc)(vector<int64_t> &, ge::Format, ge::DataType);
void (*DestroyTensorDesc)(void*);

void* (*CreateDataOp)(string);
void (*SetDataOp)(void*, void*);
void (*DestroyDataOp)(void*);

void* (*CreateConstOp)(string);
void (*SetConstOp)(void*, void*, uint8_t*, size_t);
void (*DestroyConstOp)(void*);

void* (*CreateConvTransposeOp)(string);
void (*setConvTransposeOp)(void*, void*, void*, void*,
                            vector<int64_t>, vector<int64_t>, int64_t,
                            string,vector<int64_t>);
void (*DestroyConvTransposeOp)(void*);

void* (*CreateGraph)(string);
void (*SetGraph)(void*,std::vector<void*> , std::vector<void*>);
void (*DestroyGraph)(void*);

void* (*CreateModelClient)(const std::string, void*,
                      std::vector<std::shared_ptr<hiai::AiTensor>>* ,
                      std::vector<std::shared_ptr<hiai::AiTensor>>* );
void (*DestroyModelClient)(void* client);

bool (*RunModelClient)(void*, std::string&,
                       std::vector<std::shared_ptr<hiai::AiTensor>>*,
                       std::vector<std::shared_ptr<hiai::AiTensor>>*);

int32_t (*SetTensorWithData)(std::shared_ptr<hiai::AiTensor>&, float*, uint32_t);
void (*PrintTensorDataFunc)(std::shared_ptr<hiai::AiTensor>&, int, int);



int RegistFunctions()
{
    // loading so
    void* hiaiHandler= dlopen("/data/local/tmp/libXNN_NPU.so", RTLD_LAZY);
    if(hiaiHandler == nullptr) {
        ALOGE("dynamic load failed!\n");
        return FAILED;
    }
    // loading function
    CreateTensorDesc = (void* (*)(vector<int64_t> &, ge::Format, ge::DataType))dlsym(hiaiHandler, "CreateTensorDesc");
    CHECK_SYM_RET(CreateTensorDesc);
    DestroyTensorDesc = (void (*)(void*))dlsym(hiaiHandler, "DestroyTensorDesc");
    CHECK_SYM_RET(DestroyTensorDesc);
    
    CreateDataOp = (void* (*)(string))dlsym(hiaiHandler, "CreateDataOp");
    CHECK_SYM_RET(CreateDataOp);
    SetDataOp = (void (*)(void*, void*))dlsym(hiaiHandler, "SetDataOp");
    CHECK_SYM_RET(SetDataOp);
    DestroyDataOp = (void (*)(void*))dlsym(hiaiHandler, "DestroyDataOp");
    CHECK_SYM_RET(DestroyDataOp);

    CreateConstOp = (void* (*)(string))dlsym(hiaiHandler, "CreateConstOp");
    CHECK_SYM_RET(CreateConstOp);
    SetConstOp = (void (*)(void*, void*, uint8_t*, size_t))dlsym(hiaiHandler, "SetConstOp");
    CHECK_SYM_RET(SetConstOp);
    DestroyConstOp = (void (*)(void*))dlsym(hiaiHandler, "DestroyConstOp");
    CHECK_SYM_RET(DestroyConstOp);

    CreateConvTransposeOp = (void* (*)(string))dlsym(hiaiHandler, "CreateConvTransposeOp");
    CHECK_SYM_RET(CreateConvTransposeOp);
    setConvTransposeOp = (void (*)(void*, void*, void*, void*,
                          vector<int64_t>, vector<int64_t>, int64_t,
                          string,vector<int64_t>))dlsym(hiaiHandler, "setConvTransposeOp");
    CHECK_SYM_RET(setConvTransposeOp);
    DestroyConvTransposeOp = (void (*)(void*))dlsym(hiaiHandler, "DestroyConvTransposeOp");
    CHECK_SYM_RET(DestroyConvTransposeOp);

    CreateGraph = (void* (*)(string))dlsym(hiaiHandler, "CreateGraph");
    CHECK_SYM_RET(CreateGraph);
    SetGraph = (void (*)(void*,std::vector<void*> , std::vector<void*>))dlsym(hiaiHandler, "SetGraph");
    CHECK_SYM_RET(SetGraph);
    DestroyGraph = (void (*)(void*))dlsym(hiaiHandler, "DestroyGraph");
    CHECK_SYM_RET(DestroyGraph);

    CreateModelClient = (void* (*)(const std::string, void*,
                                   std::vector<std::shared_ptr<hiai::AiTensor>>* ,
                                   std::vector<std::shared_ptr<hiai::AiTensor>>* ))
                         dlsym(hiaiHandler, "CreateModelClient");
    CHECK_SYM_RET(CreateModelClient);
    DestroyModelClient = (void (*)(void*))dlsym(hiaiHandler, "DestroyModelClient");
    CHECK_SYM_RET(DestroyModelClient);

    RunModelClient = (bool (*)(void*, std::string&,
                               std::vector<std::shared_ptr<hiai::AiTensor>>*,
                               std::vector<std::shared_ptr<hiai::AiTensor>>*))
                      dlsym(hiaiHandler, "RunModelClient");
    CHECK_SYM_RET(RunModelClient);

    SetTensorWithData = (int32_t (*)(std::shared_ptr<hiai::AiTensor>&, float* , uint32_t))dlsym(hiaiHandler, "SetTensorWithData");
    CHECK_SYM_RET(SetTensorWithData);

    PrintTensorDataFunc = (void(*)(std::shared_ptr<hiai::AiTensor>&, int, int))dlsym(hiaiHandler, "PrintTensorData");
    CHECK_SYM_RET(PrintTensorDataFunc);

    ALOGI("RegistFunctions success!");
    return SUCCESS;
}

bool BuildDeconvGraph(void* graph)
{
    ALOGI("start build graph!\n");
    auto data = CreateDataOp("data");
    vector<int64_t> dataShape = {1, 8, 864, 480};
    auto inputDesc = CreateTensorDesc(dataShape,
                                      ge::Format::FORMAT_NCHW,
                                      ge::DataType::DT_FLOAT);
    SetDataOp(data,inputDesc);

    string deconvName = "deconv";
    auto filter = CreateConstOp(deconvName + "_filter");
    {
        vector<int64_t> weightShape = {8,1,4,4};
        auto convWeightDesc = CreateTensorDesc(weightShape, ge::FORMAT_NCHW, ge::DT_FLOAT);
        vector<float> convWeightValue(128, 1);
        SetConstOp(filter, convWeightDesc, (uint8_t*)convWeightValue.data(),
                     convWeightValue.size() * sizeof(float));
    }

    auto outputShape = CreateConstOp(deconvName + "_output");
    {
        vector<int64_t> outputShapeDim = {4};
        auto outDesc = CreateTensorDesc(outputShapeDim, ge::FORMAT_NCHW, ge::DT_INT32);
        std::vector<int32_t> outShapeValue{1,1,864*2,480*2};
        SetConstOp(outputShape, outDesc,
                  (uint8_t*)outShapeValue.data(),
                   outShapeValue.size() * sizeof(int32_t));
    }

    auto bias = CreateConstOp(deconvName + "_bias");
    {
        vector<int64_t> biasShape = {1,1,1,1};
        auto convBiasDesc = CreateTensorDesc(biasShape, ge::FORMAT_NCHW, ge::DT_FLOAT);
        vector<float> convBiasValue(1, 1);
        SetConstOp(bias, convBiasDesc, (uint8_t*)convBiasValue.data(),
                     convBiasValue.size() * sizeof(float));
    }

    auto deconvOp = CreateConvTransposeOp(deconvName);
    setConvTransposeOp(deconvOp, outputShape, filter, data, {1,1}, {2,2}, 1, "SAME", {0,0,0,0});

    std::vector<void*> inputs{data};
    std::vector<void*> outputs{deconvOp};
    SetGraph(graph,inputs,outputs);
    ALOGI("finish build graph!\n");
    return true;
}

void Test1(string testName) {
    // build graph
    string modelName = testName + ".om";
    auto irGraph = CreateGraph("ir_graph");
    cout << "============= CaseName: " << testName << endl;
    BuildDeconvGraph(irGraph);

    // build model
    vector<shared_ptr<hiai::AiTensor>> inputTensors;
    vector<shared_ptr<hiai::AiTensor>> outputTensors;
    auto client = CreateModelClient(modelName, irGraph, &inputTensors, &outputTensors);
    if (client == nullptr) {
        cerr << "ERROR: build " << modelName << " failed." << endl;
        return;
    }

    // set data and run
    vector<float> inputData(1*8*864*480, 0.1);
    SetTensorWithData(inputTensors[0],inputData.data(),inputData.size()*sizeof(float));

    if (!RunModelClient(client, modelName, &inputTensors, &outputTensors)) {
        cerr << "ERROR: run " << modelName << " failed." << endl;
        return;
    }
    
    // verify
    PrintTensorDataFunc(inputTensors[0], 0, 32);
    int i = 0;
    for (shared_ptr<hiai::AiTensor>& tensor : outputTensors) {
        PrintTensorDataFunc(tensor, 0, 32);
    }
    cout << "-------------" << testName << " -------- " << endl;
}

}


int main(int argc, char* argv[]) {
    int ret = test_case::RegistFunctions();
    if(ret == FAILED) {
        return 0;
    }
    test_case::Test1("deconvOp");
    return 0;
}

