#ifndef _FACE_ANTICONV_H_
#define _FACE_ANTICONV_H_

#include <orbwebai/structure.h>
#include "../detecter.h"
#include "ncnn/net.h"

namespace mirror {
using ANCHORS = std::vector<orbwebai::Rect>;
class AntiConv : public Detecter {
public:
	AntiConv();
	~AntiConv();
	int LoadModel(const char* root_path);
	std::vector<orbwebai::face::Info> DetectFace(const orbwebai::ImageMetaInfo& img_src) override;

private:
	ncnn::Net* anticonv_net_;
	std::vector<ANCHORS> anchors_generated_;
	bool initialized_;
	const int RPNs_[3] = { 32, 16, 8 };
	const orbwebai::Size inputSize_ = { 640, 640 };
	const float iouThreshold_ = 0.4f;
	const float scoreThreshold_ = 0.8f;
    const float maskThreshold_ = 0.2f;

};

}

#endif // !_FACE_ANTICONV_H_
