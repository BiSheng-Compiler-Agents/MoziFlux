cmake_minimum_required(VERSION 3.14)
project(TRITON_VECTOR_ADD)

add_compile_options(-std=c++14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")
set(CMAKE_CXX_FLAGS_DEBUG   "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_executable(test_vector_add test_vector_add.cpp)

# CANN path (set by setenv.bash → ASCEND_HOME_PATH = .../cann-9.0.0)
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
  set(ASCEND_PATH "$ENV{HOME}/miniconda3/Ascend/cann-9.0.0")
endif()

# cann-9.0.0 layout:
#   $ASCEND_HOME_PATH/include/          -> acl/acl.h
#   $ASCEND_HOME_PATH/x86_64-linux/pkg_inc/runtime/ -> runtime/runtime/rt.h
#   $ASCEND_HOME_PATH/x86_64-linux/pkg_inc/profiling/ -> prof_common.h (needed by toolchain/prof_api.h)
include_directories(
  ${ASCEND_PATH}/include
  ${ASCEND_PATH}/include/aclnn
  ${ASCEND_PATH}/include/experiment
  ${ASCEND_PATH}/include/experiment/msprof
  ${ASCEND_PATH}/x86_64-linux/pkg_inc
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/profiling
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/runtime
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/toolchain
)

# -Wl,--allow-shlib-undefined: libruntime.so has transitive deps (liberror_manager, libmmpa)
# that are absent at link time but present in camodel at cannsim runtime.
target_link_libraries(test_vector_add PRIVATE
  ${ASCEND_PATH}/lib64/libruntime.so
)
target_link_options(test_vector_add PRIVATE -Wl,--allow-shlib-undefined)

install(TARGETS test_vector_add DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
