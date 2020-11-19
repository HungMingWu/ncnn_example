target_sources(${PROJECT_NAME}
PRIVATE
    landmarker.cpp
)

if (FACE_LANDMARKER_ALGORITHM STREQUAL "InsightFace")
target_sources(${PROJECT_NAME}
PRIVATE
    insightface/insightface.h
    insightface/insightface.cpp
)
target_compile_definitions(${PROJECT_NAME} 
PRIVATE
    USE_INSIGHTFACE
)
else()
target_sources(${PROJECT_NAME}
PRIVATE
    zqlandmarker/zqlandmarker.h
    zqlandmarker/zqlandmarker.cpp
)
target_compile_definitions(${PROJECT_NAME} 
PRIVATE
    USE_ZQLANDMARKER
)
endif()