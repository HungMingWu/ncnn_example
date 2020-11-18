#ifndef _OBJECT_DETECTOR_H_
#define _OBJECT_DETECTOR_H_

#include <vector>
#include "../common/common.h"

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
    #ifdef OBJECT_EXPORTS
        #define OBJECT_API __declspec(dllexport)
    #else
        #define OBJECT_API __declspec(dllimport)
    #endif
#else
    #define OBJECT_API __attribute__ ((visibility("default")))
#endif

namespace mirror {

class ObjectEngine {
public:
	OBJECT_API ObjectEngine();
	OBJECT_API ~ObjectEngine();

	OBJECT_API int LoadModel(const char* root_path);
	OBJECT_API std::vector<ObjectInfo> DetectObject(const mirror::ImageMetaInfo& img_src);

private:
	class Impl;
	Impl* impl_;
};

}

#endif // !_OBJECT_DETECTOR_H_

