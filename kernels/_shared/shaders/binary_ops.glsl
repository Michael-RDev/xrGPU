// SPDX-License-Identifier: Apache-2.0
// Original native Vulkan arithmetic. Packed narrow values are widened to uint;
// arithmetic results are narrowed modulo their storage width after evaluation.
uint evaluateBinary(uint a, uint b) {
#if DTYPE == 0 || DTYPE == 1
  float x = decodeFloat(a), y = decodeFloat(b);
#if OPCODE == 0
  return encodeFloat(x + y);
#elif OPCODE == 3
  return encodeFloat(x - y);
#elif OPCODE == 4
  return encodeFloat(x * y);
#elif OPCODE == 5
  return encodeFloat(x / y);
#elif OPCODE == 9
  return encodeFloat(x < 0.0 ? x * y : x);
#else
#error Invalid floating binary operation
#endif
#else
#if OPCODE == 0
  return (a + b) & MASK;
#elif OPCODE == 3
  return (a - b) & MASK;
#elif OPCODE == 4
  return (a * b) & MASK;
#elif OPCODE == 5
  // Define the otherwise undefined SPIR-V division cases without evaluating
  // the invalid division. Narrow signed inputs are explicitly sign-extended.
#if DTYPE == 2 || DTYPE == 4 || DTYPE == 6
#if DTYPE == 4
  int x = int(a << 24u) >> 24;
  int y = int(b << 24u) >> 24;
#elif DTYPE == 6
  int x = int(a << 16u) >> 16;
  int y = int(b << 16u) >> 16;
#else
  int x = int(a), y = int(b);
#endif
  if (y == 0) return a;
  if (a == 0x80000000u && y == -1) return a;
  return uint(x / y) & MASK;
#else
  if (b == 0u) return a;
  return (a / b) & MASK;
#endif
#elif OPCODE == 6
  return (a & b) & MASK;
#elif OPCODE == 7
  return (a | b) & MASK;
#elif OPCODE == 8
  return (a ^ b) & MASK;
#else
#error Invalid integral binary operation
#endif
#endif
}
