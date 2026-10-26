#include <atomic>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>
#include <xrgpu/session.h>
using Json = nlohmann::json;
void check(bool condition) {
  if (!condition)
    throw std::runtime_error("Session contract check failed");
}
std::vector<uint8_t> bytes(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Missing fixture");
  return {std::istreambuf_iterator<char>(stream), {}};
}
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("Session test requires a model bundle");
    const std::filesystem::path root = argv[1];
    std::ifstream stream(root / "bundle.json");
    const auto bundle = Json::parse(stream);
    std::vector<xrgpu::Tensor> inputs;
    for (const auto &spec : bundle.at("fixtures").at(0).at("inputs")) {
      xrgpu::Tensor tensor;
      tensor.name = spec.at("name");
      tensor.shape = spec.at("shape").get<std::vector<int64_t>>();
      tensor.dtype = spec.at("dtype") == "float32" ? xrgpu::DType::Float32
                                                   : xrgpu::DType::Int64;
      tensor.data = bytes(root / spec.at("file").get<std::string>());
      inputs.push_back(std::move(tensor));
    }
    std::unique_ptr<xrgpu::Session> loaded[2];
    std::exception_ptr errors[2];
    auto load = [&](size_t index) {
      try {
        loaded[index] = xrgpu::Session::load(root.string());
      } catch (...) {
        errors[index] = std::current_exception();
      }
    };
    std::thread first_load(load, 0), second_load(load, 1);
    first_load.join();
    second_load.join();
    for (const auto &error : errors)
      if (error)
        std::rethrow_exception(error);
    auto session = std::move(loaded[0]);
    auto other = std::move(loaded[1]);
    check(Json::parse(session->info()).at("backend") == "cpu");
    auto first = session->forward(inputs);
    const auto saved = first;
    auto again = session->forward(inputs);
    check(first.size() == bundle.at("outputs").size());
    check(first.size() == again.size());
    for (size_t i = 0; i < first.size(); ++i) {
      check(first[i].data == again[i].data);
      check(first[i].name == bundle.at("outputs")[i].at("name"));
    }
    auto invalid = inputs;
    invalid[0].name = "wrong";
    try {
      session->forward(invalid);
      throw std::runtime_error("Invalid tensor accepted");
    } catch (const std::invalid_argument &) {
    }
    invalid = inputs;
    invalid[0].data.pop_back();
    try {
      session->forward(invalid);
      throw std::runtime_error("Invalid byte count accepted");
    } catch (const std::invalid_argument &) {
    }
    invalid = inputs;
    invalid[0].dtype = inputs[0].dtype == xrgpu::DType::Float32
                           ? xrgpu::DType::Int64
                           : xrgpu::DType::Float32;
    try {
      session->forward(invalid);
      throw std::runtime_error("Invalid dtype accepted");
    } catch (const std::invalid_argument &) {
    }
    std::atomic<bool> valid{true};
    auto run = [&] {
      try {
        auto output = session->forward(inputs);
        if (output[0].data != saved[0].data)
          valid = false;
      } catch (...) {
        valid = false;
      }
    };
    std::thread a(run), b(run);
    a.join();
    b.join();
    check(valid);
    session->close();
    session->close();
    check(Json::parse(session->info()).at("closed") == true);
    try {
      session->forward(inputs);
      throw std::invalid_argument("Closed session accepted inputs");
    } catch (const std::runtime_error &) {
    }
    auto independent = other->forward(inputs);
    check(independent[0].data == saved[0].data);
    check(first[0].data == saved[0].data);
    xrgpu::SessionOptions candidate;
    candidate.experiment = "linear:m4_wg8x4_buffer";
    try {
      auto prohibited = xrgpu::Session::load(root.string(), candidate);
      throw std::runtime_error("Production accepted candidate override");
    } catch (const std::invalid_argument &) {
    }
    std::cout << "Native Session lifecycle, tensor, ownership, isolation and "
                 "concurrency checks passed"
              << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
