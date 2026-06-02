cmake_minimum_required(VERSION 3.14)
project(TRITON_FUSED_SOFTMAX)

add_compile_options(-std=c++14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")
set(CMAKE_CXX_FLAGS_DEBUG   "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_executable(test_fused_softmax test_fused_softmax.cpp)

# CANN path — set by setenv.bash via ASCEND_HOME_PATH
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
  set(ASCEND_PATH "$ENV{HOME}/miniconda3/Ascend/cann-9.0.0")
endif()

include_directories(
  ${ASCEND_PATH}/include
  ${ASCEND_PATH}/include/aclnn
  ${ASCEND_PATH}/include/experiment
  ${ASCEND_PATH}/include/experiment/msprof
  ${ASCEND_PATH}/x86_64-linux/pkg_inc
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/profiling
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/toolchain
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/runtime
)

# Link ONLY libruntime.so — do NOT link libascendcl.so
# cannsim swaps libruntime.so → libruntime_camodel.so via LD_LIBRARY_PATH prepend
# libascendcl.so is NOT in the camodel dir so aclInit would hit the real driver
# -Wl,--allow-shlib-undefined: libruntime.so transitive deps absent at link time
target_link_libraries(test_fused_softmax PRIVATE
  ${ASCEND_PATH}/lib64/libruntime.so
)
target_link_options(test_fused_softmax PRIVATE -Wl,--allow-shlib-undefined)

install(TARGETS test_fused_softmax DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
