if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "EmbedOptixPtx.cmake requires INPUT and OUTPUT paths")
endif()
file(READ "${INPUT}" _ptx)
string(FIND "${_ptx}" ")granit_mochi_ptx\"" _delimiter_position)
if(NOT _delimiter_position EQUAL -1)
    message(FATAL_ERROR "PTX contains the reserved C++ raw-string delimiter")
endif()
file(WRITE "${OUTPUT}"
    "#pragma once\n\nnamespace cudadem {\nnamespace rt {\n"
    "inline constexpr char mochi_ptx[] = R\"granit_mochi_ptx(${_ptx})granit_mochi_ptx\";\n"
    "} // namespace rt\n} // namespace cudadem\n")
