cmake_minimum_required(VERSION 3.10)
project(test_vector_add CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
  set(ASCEND_PATH "/opt/miniconda3/Ascend/cann")
endif()

include_directories(
  ${ASCEND_PATH}/include
  ${ASCEND_PATH}/x86_64-linux/pkg_inc
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/runtime
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/profiling
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/toolchain
)

add_executable(test_vector_add ${CMAKE_CURRENT_SOURCE_DIR}/test_vector_add.cpp)
target_link_libraries(test_vector_add PRIVATE ${ASCEND_PATH}/lib64/libruntime.so)
target_link_options(test_vector_add PRIVATE -Wl,--allow-shlib-undefined)
