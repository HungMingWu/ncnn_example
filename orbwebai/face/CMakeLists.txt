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
    tracker.cpp
    alignface.cpp
)

add_subdirectory(database)
add_subdirectory(detecter)
add_subdirectory(landmarker)
add_subdirectory(recognizer)