#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "../classifier_engine.h"

namespace mirror {

class Classifier {
public:
	virtual ~Classifier() {}
	virtual int LoadModel(const char* root_path) = 0;
	virtual std::vector<ImageInfo> Classify(const cv::Mat& img_src) = 0;
};



}




#endif // !_CLASSIFIER_H_

