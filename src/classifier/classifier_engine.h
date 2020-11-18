#ifndef _CLASSIFIER_ENGINE_H_
#define _CLASSIFIER_ENGINE_H_

#include <iostream>
#include <string>
#include <vector>
#include <orbwebai/structure.h>

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
    #ifdef CLASSIFIER_EXPORTS
        #define CLASSIFIER_API __declspec(dllexport)
    #else
        #define CLASSIFIER_API __declspec(dllimport)
    #endif
#else
    #define CLASSIFIER_API __attribute__ ((visibility("default")))
#endif

namespace mirror {
class ClassifierEngine {
public:
	CLASSIFIER_API ClassifierEngine();
	CLASSIFIER_API ~ClassifierEngine();
	CLASSIFIER_API int LoadModel(const char* root_path);
	CLASSIFIER_API std::vector<orbwebai::classify::Info> Classify(const orbwebai::ImageMetaInfo& img_src);

private:
	class Impl;
	Impl* impl_;

};

}

#endif // !_CLASSIFIER_ENGINE_H_

