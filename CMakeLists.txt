cmake_minimum_required (VERSION 3.0)
project (tensorflow_c_warp)
set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(TFWrapper SHARED src/FileUtils.cpp src/TFModelUnit.cpp src/StatusSingleton.cpp src/TFUtils.cpp)
link_directories(${CMAKE_CURRENT_SOURCE_DIR})


add_library(load_current_model SHARED test_load_chien_de_mer_model.cpp)
add_dependencies(load_current_model TFWrapper)
target_link_libraries(load_current_model TFWrapper)
