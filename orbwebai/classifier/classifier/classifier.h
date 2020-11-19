#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <vector>
#include <orbwebai/structure.h>

namespace mirror {

class Classifier {
public:
	virtual ~Classifier() {}
	virtual int LoadModel(const char* root_path) = 0;
	virtual std::vector<orbwebai::classify::Info> Classify(const orbwebai::ImageMetaInfo& img_src) = 0;
};



}




#endif // !_CLASSIFIER_H_

