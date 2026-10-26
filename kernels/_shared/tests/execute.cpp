// SPDX-License-Identifier: Apache-2.0
// Raw tensor executable used only by the independent local conformance harness.
#include <xrgpu/operators.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace xrgpu::operators;
int main(int argc, char** argv) {
  try {
    if(argc<8 || (argc-5)%3) throw std::invalid_argument("PACKAGE OP DTYPE OUTPUT_PREFIX [NAME SHAPE FILE]...");
    std::vector<Tensor> inputs;
    for(int i=5;i<argc;i+=3) {
      Tensor tensor;tensor.name=argv[i];tensor.dtype=parse_dtype(argv[3]);
      if(std::string(argv[i+1])!="scalar") {
        std::istringstream shape(argv[i+1]);std::string item;
        while(std::getline(shape,item,','))tensor.shape.push_back(std::stoll(item));
      }
      std::ifstream file(argv[i+2],std::ios::binary|std::ios::ate);
      if(!file || file.tellg()<0)throw std::runtime_error("Missing input bytes");
      tensor.data.resize(static_cast<size_t>(file.tellg()));file.seekg(0);
      file.read(reinterpret_cast<char*>(tensor.data.data()),tensor.data.size());
      if(!file)throw std::runtime_error("Incomplete input bytes");
      inputs.push_back(std::move(tensor));
    }
    auto operation=Operator::prepare(argv[1],argv[2],inputs);
    operation->run();auto outputs=operation->read_outputs();
    for(size_t i=0;i<outputs.size();++i) {
      std::ofstream file(std::string(argv[4])+"."+std::to_string(i)+".bin",std::ios::binary);
      file.write(reinterpret_cast<const char*>(outputs[i].data.data()),outputs[i].data.size());
      if(!file)throw std::runtime_error("Could not write output bytes");
    }
    std::cout<<operation->info()<<'\n';
    operation->close();return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
