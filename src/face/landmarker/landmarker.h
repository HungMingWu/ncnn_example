#ifndef _FACE_LANDMARKER_H_
#define _FACE_LANDMARKER_H_

#include "opencv2/core.hpp"

namespace mirror {
// 抽象类
class Landmarker {
public:
	virtual ~Landmarker() {};
	virtual int LoadModel(const char* root_path) = 0;
	virtual std::vector<cv::Point2f> ExtractKeypoints(const cv::Mat& img_src,
		const cv::Rect& face) = 0;
};

}

#endif // !_FACE_LANDMARKER_H_

