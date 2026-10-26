// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>
namespace xrgpu::operators::detail {
enum class Kind { Unary, Binary, MatMul };
struct OperationRecord { const char* id; const char* slug; Kind kind; uint32_t opcode; uint32_t dtype_mask; const char* input_a; const char* input_b; const char* output; };
inline constexpr OperationRecord kOperations[] = {
#define XRGPU_OPERATOR(id, slug, kind, opcode, mask, a, b, output) {id, slug, Kind::kind, opcode, mask, a, b, output},
#include "registry.def"
#if __has_include("registry_binary.def")
#include "registry_binary.def"
#endif
#if __has_include("registry_unary.def")
#include "registry_unary.def"
#endif
#undef XRGPU_OPERATOR
};
inline const OperationRecord& operation_record(const std::string& id) {
 for(const auto& record:kOperations)if(id==record.id)return record;
 throw std::invalid_argument("Unsupported native operator: "+id);
}
}
