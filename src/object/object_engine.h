#ifndef _OBJECT_DETECTOR_H_
#define _OBJECT_DETECTOR_H_

#include <vector>
#include <orbwebai/structure.h>

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
	OBJECT_API std::vector<orbwebai::object::Info> DetectObject(const orbwebai::ImageMetaInfo& img_src);

private:
	class Impl;
	Impl* impl_;
};

}

#endif // !_OBJECT_DETECTOR_H_

