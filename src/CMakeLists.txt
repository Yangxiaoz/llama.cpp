llama_add_compile_flags()

#custom add:
# check liburing
find_library(LIBURING_LIBRARY NAMES uring)
if(NOT LIBURING_LIBRARY)
    message(FATAL_ERROR "liburing 未找到 - 请安装: sudo apt install liburing-dev")
endif()
find_path(LIBURING_INCLUDE_DIR NAMES liburing.h)
#end

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache-unified.cpp
            llama-kv-cache-unified-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            custom-moe-offload.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            )
#custom add
# target_include_directories(llama PRIVATE ${LIBURING_LIBRARY})
if(LIBURING_INCLUDE_DIR AND LIBURING_LIBRARY)
    message(STATUS "liburing found")
    target_include_directories(llama PRIVATE ${LIBURING_INCLUDE_DIR})
    target_link_libraries(llama PRIVATE ${LIBURING_LIBRARY})
    target_compile_definitions(llama PRIVATE LLAMA_USE_LIBURING)
else()
    message(WARNING "liburing not found - custom-moe-offload.cpp will be compiled without io_uring support")
endif()
#end

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
