cmake_minimum_required(VERSION 3.13)
project(orbwebai)

Option(MIRROR_OPENMP "openmp support" ON)
Option(MIRROR_VULKAN "vulkan compute used" OFF)

Option(BUILD_FACE_ENGINE "Build face engine" ON)

# FACE_DETECT_ALGORITHM should be "RetinaFace", "MTCNN", "CenterFace", "AntiConv"
set(FACE_DETECT_ALGORITHM "CenterFace" CACHE STRING "face Detect Algorithm")

option(BUILD_FACE_LANDMARKER "Build face engine landmakrer" ON)

# FACE_LANDMARKER_ALGORITHM should be "InsightFace", "zqlandmarker"
set(FACE_LANDMARKER_ALGORITHM "zqlandmarker" CACHE STRING "face Detect Algorithm")

option(BUILD_FACE_RECOGNIZER "Build face engine recognizer" ON)
option(BUILD_FACE_ALIGN "Build face engine align" ON)
option(BUILD_FACE_DATABASE "Build face engine database" ON)
option(BUILD_FACE_TRACKER "Build face engine tracker" ON)
option(BUILD_OBJECT_ENGINE "Build object engine" ON)
option(BUILD_CLASSIFIER_ENGINE "Build classifier engine" ON)

if (MIRROR_OPENMP)
	find_package(OpenMP)
endif()

if (BUILD_FACE_ALIGN AND NOT BUILD_FACE_LANDMARKER)
	message("Face align needs landmarker, disable it")
	set(BUILD_FACE_ALIGN OFF)
endif()

add_subdirectory(orbwebai)
add_subdirectory(examples)
