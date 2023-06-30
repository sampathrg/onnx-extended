#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>

template<typename T, int32_t Rank>
struct Memref {
  T *bufferPtr;
  T *alignedPtr;
  int64_t offset;
  int64_t lengths[Rank];
  int64_t strides[Rank];
};



typedef int32_t (*InitModelFn)();

class TreebeardSORunner {
  void* so;
  void* predFnPtr;
  int32_t batchSize;
  int32_t rowSize;

  void CallFuncAndGetIntValueFromSo(const std::string& functionName, int32_t& field) {
    using GetFunc_t = int32_t(*)();
    auto get = reinterpret_cast<GetFunc_t>(dlsym(so, functionName.c_str()));
    field = get();
  }
public:
  TreebeardSORunner(const char *soFilePath) {
    so = dlopen(soFilePath, RTLD_NOW);
    if (!so) {
      std::cout << "Failed to load so: " << soFilePath << std::endl;
      throw std::runtime_error("Failed to load so");
    }
    auto initModelFnPtr = (InitModelFn)dlsym(so, "Init_model");
    if (!initModelFnPtr) {
      std::cout << "Failed to load Init_model function from so: " << soFilePath << std::endl;
      throw std::runtime_error("Failed to load Init_model function from so");
    }
    initModelFnPtr();
    CallFuncAndGetIntValueFromSo("GetBatchSize", batchSize);
    CallFuncAndGetIntValueFromSo("GetRowSize", rowSize);
    
    predFnPtr = dlsym(so, "Prediction_Function");
  }

  ~TreebeardSORunner() {
    dlclose(so);
  }

  int32_t GetBatchSize() const { return batchSize; }
  int32_t GetRowSize() const { return rowSize; }

  template<typename InputElementType, typename ReturnType>
  int32_t RunInference(InputElementType *input, ReturnType *returnValue) {
    typedef Memref<ReturnType, 1> (*InferenceFunc_t)(
        InputElementType*, InputElementType*, int64_t, int64_t, int64_t, int64_t, int64_t,
        ReturnType*, ReturnType*, int64_t, int64_t, int64_t);
    auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(predFnPtr);

    InputElementType *ptr = input;
    InputElementType *alignedPtr = input;

    ReturnType *resultPtr = returnValue;
    ReturnType *resultAlignedPtr = returnValue;

    int64_t offset = 0, stride = 1;
    int64_t resultLen = batchSize;

    inferenceFuncPtr(ptr, alignedPtr, offset, batchSize, rowSize, rowSize, stride,
                     resultPtr, resultAlignedPtr, offset, resultLen, stride);
    return 0;
  }
};