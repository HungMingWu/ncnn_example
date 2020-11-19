project(VisionEngine)

add_library(${PROJECT_NAME} STATIC "")

set_target_properties(${PROJECT_NAME}
PROPERTIES
    CXX_STANDARD 11
)

target_link_directories(${PROJECT_NAME}
PUBLIC
    ../../third_party/ncnn/lib/win
    ../../third_party/ncnn/lib/linux
)

target_link_libraries(${PROJECT_NAME}
PRIVATE
    ncnn
    $<$<AND:$<BOOL:${MIRROR_OPENMP}>,$<BOOL:${OpenMP_CXX_FOUND}>>:OpenMP::OpenMP_CXX>
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ../../include
PRIVATE
    ../common/include
    ../../third_party/eigen/
    ../../third_party/ncnn/include
)

target_sources(${PROJECT_NAME}
PRIVATE
    ../common/src/common.cpp
    $<$<BOOL:${BUILD_FACE_ALIGN}>:alignface.cpp>
    $<$<BOOL:${BUILD_FACE_TRACKER}>:tracker.cpp>
)

add_subdirectory(detecter)

if (BUILD_FACE_DATABASE)
    add_subdirectory(database)
endif()

if (BUILD_FACE_LANDMARKER)
    add_subdirectory(landmarker)
endif()

if (BUILD_FACE_RECOGNIZER)
    add_subdirectory(recognizer)
endif()
