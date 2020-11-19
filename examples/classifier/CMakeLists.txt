add_executable(test_classifier test_classifier.cpp)
target_link_libraries(test_classifier
PRIVATE
    ClassifierEngine
    ${OpenCV_LIBS}
)
