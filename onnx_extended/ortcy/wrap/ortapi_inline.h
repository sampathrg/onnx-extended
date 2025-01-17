#pragma once

#include "helpers.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_c_api.h"
#undef ORT_API_MANUAL_INIT

namespace ortapi {

inline static const OrtApi *GetOrtApi() { 
    const OrtApi* api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return api_;
}

inline const char* ort_version() { return OrtGetApiBase()->GetVersionString(); }

inline void _ThrowOnError_(OrtStatus* ort_status, const char* filename, int line) {
    if (ort_status) {
        std::string message(GetOrtApi()->GetErrorMessage(ort_status));
        OrtErrorCode code = GetOrtApi()->GetErrorCode(ort_status);
        throw std::runtime_error(
            orthelpers::MakeString("error: onnxruntime(", code, "), ", message, "\n    ", filename, ":", line));
    }
}

#define ThrowOnError(ort_status) _ThrowOnError_(ort_status, __FILE__, __LINE__)

} // namespace ortapi
