add_executable(face_detect face_detect.cpp)
target_link_libraries(face_detect
PRIVATE
    VisionEngine
    ${OpenCV_LIBS}
)

add_executable(test_face test_face.cpp)
target_link_libraries(test_face
PRIVATE
    VisionEngine
    ${OpenCV_LIBS}
)

