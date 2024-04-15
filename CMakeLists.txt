cmake_minimum_required(VERSION 3.10)
project(KalmanFilterExample)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 寻找 Eigen 库
find_package(Eigen3 REQUIRED)

# 添加可执行文件
add_executable(kalman_filter kallman.cpp)

# 将 Eigen 头文件包含到项目中
target_include_directories(kalman_filter PRIVATE ${EIGEN3_INCLUDE_DIRS})

