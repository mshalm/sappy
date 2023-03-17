cmake_minimum_required(VERSION 3.0.0)
project(sappy_solver VERSION 0.1.0)
if (APPLE)
set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")
endif()
add_subdirectory(pybind11)
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/eigen/cmake")
find_package(Eigen3 3.3.90 EXACT REQUIRED)
set(SOURCE_DIR "src")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(SRCS
    ${SOURCE_DIR}/Solver.cpp
    )
    
set(HEADERS
    ${SOURCE_DIR}/Solver.hpp
    )

include_directories(${EIGEN3_INCLUDE_DIR})

add_library(sappylib ${SRCS} ${HEADERS})
target_link_libraries(sappylib pybind11::pybind11)
target_compile_options(sappylib PRIVATE -fPIC)

pybind11_add_module(sappy_solver "${SOURCE_DIR}/main.cpp")
target_link_libraries(sappy_solver PRIVATE sappylib)
target_compile_options(sappy_solver PRIVATE -fPIC)

