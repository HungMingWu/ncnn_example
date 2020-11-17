#ifndef _FACE_ALIGNER_H_
#define _FACE_ALIGNER_H_

#include "opencv2/core.hpp"
#include "common.h"

namespace mirror {
class Aligner {
public:
	Aligner();
	~Aligner();

	int AlignFace(const cv::Mat & img_src,
		const std::vector<mirror::Point2f>& keypoints, cv::Mat * face_aligned);

private:
	class Impl;
	Impl* impl_;
};

}

#endif // !_FACE_ALIGNER_H_

