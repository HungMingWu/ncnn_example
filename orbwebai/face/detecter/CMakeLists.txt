target_sources(${PROJECT_NAME}
PRIVATE
    detecter.cpp
)
if (FACE_DETECT_ALGORITHM STREQUAL "RetinaFace")
target_sources(${PROJECT_NAME}
PRIVATE
    retinaface/retinaface.h
    retinaface/retinaface.cpp
)
target_compile_definitions(${PROJECT_NAME} 
PRIVATE
    USE_RETIMAFCE
)
elseif (FACE_DETECT_ALGORITHM STREQUAL "MTCNN")
target_sources(${PROJECT_NAME}
PRIVATE
    mtcnn/mtcnn.h
    mtcnn/mtcnn.cpp
)
target_compile_definitions(${PROJECT_NAME} 
PRIVATE
    USE_MTCNN
)
elseif (FACE_DETECT_ALGORITHM STREQUAL "CenterFace")
target_sources(${PROJECT_NAME}
PRIVATE
    centerface/centerface.h
    centerface/centerface.cpp
)
target_compile_definitions(${PROJECT_NAME} 
PRIVATE
    USE_CENTERFACE
)
else()
target_sources(${PROJECT_NAME}
PRIVATE
    anticonv/anticonv.h
    anticonv/anticonv.cpp
)
target_compile_definitions(${PROJECT_NAME} 
PRIVATE
    USE_ANTICONV
)
endif()
