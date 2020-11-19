add_executable(face_detect face_detect.cpp)
target_link_libraries(face_detect
PUBLIC
    VisionEngine
PRIVATE
    ${OpenCV_LIBS}
)
