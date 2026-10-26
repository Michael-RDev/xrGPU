#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <xrgpu/session.h>

int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument("Usage: xrgpu_run BUNDLE [INPUT.bin ...]");
    auto session = xrgpu::Session::load(argv[1]);
    auto info = nlohmann::json::parse(session->info());
    if (argc == 2) {
      std::cout << info.dump() << std::endl;
      return 0;
    }
    const auto &specs = info.at("inputs");
    if (static_cast<size_t>(argc - 2) != specs.size())
      throw std::invalid_argument("Input file count differs from model");
    std::vector<xrgpu::Tensor> inputs;
    for (size_t i = 0; i < specs.size(); ++i) {
      std::ifstream stream(argv[i + 2], std::ios::binary);
      if (!stream)
        throw std::invalid_argument("Cannot read input file");
      xrgpu::Tensor tensor;
      tensor.name = specs[i].at("name");
      tensor.shape = specs[i].at("shape").get<std::vector<int64_t>>();
      tensor.dtype = specs[i].at("dtype") == "float32" ? xrgpu::DType::Float32
                                                       : xrgpu::DType::Int64;
      tensor.data =
          std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), {});
      inputs.push_back(std::move(tensor));
    }
    auto outputs = session->forward(inputs);
    info = nlohmann::json::parse(session->info());
    info["output_files"] = nlohmann::json::array();
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto name = "output" + std::to_string(i) + ".bin";
      std::ofstream stream(name, std::ios::binary);
      stream.write(reinterpret_cast<const char *>(outputs[i].data.data()),
                   outputs[i].data.size());
      if (!stream)
        throw std::runtime_error("Cannot write output");
      info["output_files"].push_back(name);
    }
    session->close();
    std::cout << info.dump() << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
