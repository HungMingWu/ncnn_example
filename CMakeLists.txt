cmake_minimum_required(VERSION 3.13)
project(orbwebai)

Option(MIRROR_OPENMP "openmp support" ON)
Option(MIRROR_VULKAN "vulkan compute used" OFF)
Option(MIRROR_INSTALL_SDK "install mirror library and headers" ON)

option(BUILD_FACE_ENGINE "Build face engine" ON)
option(BUILD_OBJECT_ENGINE "Build object engine" ON)
option(BUILD_CLASSIFIER_ENGINE "Build classifier engine" ON)
add_subdirectory(orbwebai)
add_subdirectory(examples)