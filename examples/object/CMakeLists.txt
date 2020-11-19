add_executable(test_object test_object.cpp)
target_link_libraries(test_object
PRIVATE
    ObjectEngine
    ${OpenCV_LIBS}
)
