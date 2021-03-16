// 此处包含C++头文件
#include<iostream>
#include"adapter.h"
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif
void* CreateDataOp(string name) {
    auto ptr = new ge::op::Data(name);
    return (void*)ptr;
}

void SetDataOp(void* dataOp,void* tensorDesc){
    ((ge::op::Data*)dataOp)->update_input_desc_x(*(ge::TensorDesc *)tensorDesc);
}

void DestroyDataOp(void* ptr) {
    ge::op::Data* obj = (ge::op::Data*)ptr;
    delete obj;
}

void* CreateConstOp(string name) {
    auto ptr = new hiai::op::Const(name);
    return (void*)ptr;
}

void SetConstOp(void* op, void* wDesc, uint8_t* data, size_t dataSize) {
    hiai::op::Const* constOp = (hiai::op::Const*)op;
    hiai::TensorPtr weight = std::make_shared<hiai::Tensor>();
    weight->SetTensorDesc(*(hiai::TensorDesc*)wDesc);
    weight->SetData(data, dataSize);
    constOp->set_attr_value(weight);
}

void DestroyConstOp(void* ptr) {
    hiai::op::Const* obj = (hiai::op::Const*)ptr;
    delete obj;
}

void* CreateConvTransposeOp(string name) {
    auto ptr = new hiai::op::ConvTranspose(name);
    return (void*)ptr;
}

void setConvTransposeOp(void* op, void* outputshape, void* filter, void* input,
                        vector<int64_t> dialations, vector<int64_t> strides, int64_t group,
                        string padmode,vector<int64_t> pads) {
    auto deconvOp = (hiai::op::ConvTranspose*)op;
    (*deconvOp).set_input_output_shape(*(hiai::op::Const*)outputshape)
                .set_input_filter(*(hiai::op::Const*)filter)
                .set_input_x(*(ge::Operator*)input)
                .set_attr_dilations({dialations[0], dialations[1]})
                .set_attr_strides({strides[0], strides[1]})
                .set_attr_groups(group)
                .set_attr_pad_mode(padmode)
                .set_attr_pads({pads[0],pads[1],pads[2],pads[3]});
    
}

void DestroyConvTransposeOp(void* ptr) {
    hiai::op::ConvTranspose* obj = (hiai::op::ConvTranspose*)ptr;
    delete obj;
}

#ifdef __cplusplus
}
#endif