#ifndef _FACE_DETECTER_H_
#define _FACE_DETECTER_H_

#include "opencv2/core.hpp"
#include "../common/common.h"

namespace mirror {
// 抽象类
class Detecter {
public:
	virtual ~Detecter() {};
	virtual int LoadModel(const char* root_path) = 0;
	virtual std::vector<FaceInfo> DetectFace(const cv::Mat& img_src) = 0;

};

}

#endif // !_FACE_DETECTER_H_

