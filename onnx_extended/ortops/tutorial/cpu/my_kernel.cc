#include "my_kernel.h"
#include <sstream>

namespace ortops {

MyCustomKernel::MyCustomKernel(const OrtApi &api, const OrtKernelInfo *info) : soRunner("/home/srajendra/temp/rf_nf10_T500_d10.onnx.so") {
}

void MyCustomKernel::Compute(OrtKernelContext *context) {
  Ort::KernelContext ctx(context);

  Ort::ConstValue input_X = ctx.GetInput(0);
  const float *X = input_X.GetTensorData<float>();

  // Setup output, which is assumed to have the same dimensions as the inputs.
  std::vector<int64_t> dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();

  if (dimensions[0] != soRunner.GetBatchSize() || dimensions[1] != soRunner.GetRowSize()) {
    std::stringstream ss;
    ss << "Dimensions not " << soRunner.GetBatchSize() << " x " << soRunner.GetRowSize();
    throw std::runtime_error(ss.str());
  }

  Ort::UnownedValue output = ctx.GetOutput(0, {dimensions[0], 1});
  float *out = output.GetTensorMutableData<float>();

  const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();
  soRunner.RunInference(X, out);
}

void* MyCustomOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return std::make_unique<MyCustomKernel>(api, info).release();
};

const char* MyCustomOp::GetName() const { return "MyCustomOp"; };

const char* MyCustomOp::GetExecutionProviderType() const { return "CPUExecutionProvider"; };

size_t MyCustomOp::GetInputTypeCount() const { return 1; };

ONNXTensorElementDataType MyCustomOp::GetInputType(size_t index) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

size_t MyCustomOp::GetOutputTypeCount() const { return 1; };

ONNXTensorElementDataType MyCustomOp::GetOutputType(size_t index) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

} // namespace ortops
