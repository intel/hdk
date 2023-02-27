#pragma once
#include <string>
namespace compiler {
  enum class CallingConvDesc{NONE=0, C, SPIR_FUNC};
  struct CodegenTraitsDescriptor {
    CodegenTraitsDescriptor(unsigned local_addr_space,
                            unsigned global_addr_space,
                            CallingConvDesc calling_conv,
                            std::string_view triple)
        : local_addr_space_(local_addr_space)
        , global_addr_space_(global_addr_space)
        , conv_(calling_conv)
        , triple_(triple) {}
    CodegenTraitsDescriptor(){};

    unsigned local_addr_space_{0};
    unsigned global_addr_space_{0};
    CallingConvDesc conv_{CallingConvDesc::NONE};
    std::string_view triple_{"DUMMY"};
  };
}