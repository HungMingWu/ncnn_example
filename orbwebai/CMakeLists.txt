if (BUILD_FACE_ENGINE)
	add_subdirectory(face)
endif()

if (BUILD_OBJECT_ENGINE)
	add_subdirectory(object)
endif()

if (BUILD_CLASSIFIER_ENGINE)
	add_subdirectory(classifier)
endif()