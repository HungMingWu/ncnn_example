add_executable(face_detect face_detect.cpp)
target_link_libraries(face_detect
PRIVATE
    VisionEngine
    ${OpenCV_LIBS}
)

add_executable(test_mask test_mask.cpp)
target_link_libraries(test_mask
PRIVATE
    VisionEngine
    ${OpenCV_LIBS}
)

if (BUILD_FACE_DATABASE)
add_executable(test_database test_database.cpp)
target_link_libraries(test_database
PRIVATE
    VisionEngine
    ${OpenCV_LIBS}
)
endif()

if (BUILD_FACE_LANDMARKER)
add_executable(test_landmarker test_landmarker.cpp)
target_link_libraries(test_landmarker
PRIVATE
    VisionEngine
    ${OpenCV_LIBS}
)
endif()

if (BUILD_FACE_ALIGN)
add_executable(test_align test_align.cpp)
target_link_libraries(test_align
PRIVATE
    VisionEngine
    ${OpenCV_LIBS}
)
endif()

if (BUILD_FACE_RECOGNIZER)
add_executable(test_recognizer test_recognizer.cpp)
target_link_libraries(test_recognizer
PRIVATE
    VisionEngine
    ${OpenCV_LIBS}
)
endif()

if (BUILD_FACE_TRACKER)
add_executable(test_track test_track.cpp)
target_link_libraries(test_track
PRIVATE
    VisionEngine
    ${OpenCV_LIBS}
)
endif()