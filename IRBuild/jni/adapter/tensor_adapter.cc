// 此处包含C++头文件
#include<iostream>
#include"adapter.h"
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

void* CreateTensorDesc(vector<int64_t> &shape, ge::Format format, ge::DataType dt) {
    auto inputDesc = new ge::TensorDesc(ge::Shape(shape),format,dt);
    return (void*)inputDesc;
}

void DestroyTensorDesc(void* ptr) {
    ge::TensorDesc* obj = (ge::TensorDesc*)ptr;
    delete obj;
}

int32_t SetTensorWithData(std::shared_ptr<hiai::AiTensor>& tensor, float* dataPtr, uint32_t dataSize) {
    auto size = tensor->GetSize();
    if (size != dataSize) {
        ALOGE("tensor size(%u) != dataSize(%u)\n", size, dataSize);
        return FAILED;
    }
    (void)memcpy(tensor->GetBuffer(), dataPtr, dataSize);
    return SUCCESS;
}

void PrintTensorData(std::shared_ptr<hiai::AiTensor>& tensor, int start, int end) {
    const int printNumOnEachLine = 16;
    auto size = tensor->GetSize();
    int num = size / sizeof(float);
    auto ptr = (float*)tensor->GetBuffer();

    std::cout << "Print tensor " << tensor << " size: " << size << " num: " << num;
    for (int i = start; i < std::min(num, end); i++) {
        if (i % printNumOnEachLine == 0) {
            std::cout << std::endl
                      << "[" << std::setfill('0') << std::setw(5) << i << ", "
                      << std::setfill('0') << std::setw(5)
                      << std::min(i + printNumOnEachLine - 1, end) << "]";
        }
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << (float)ptr[i] << " ";
    }
    std::cout << std::endl;
}

#ifdef __cplusplus
}
#endif