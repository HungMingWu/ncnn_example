#ifndef _FACE_ALIGNER_H_
#define _FACE_ALIGNER_H_

#include "common.h"

namespace mirror {
class Aligner {
public:
	Aligner();
	~Aligner();

	int AlignFace(const mirror::ImageMetaInfo& img_src,
		const std::vector<mirror::Point2f>& keypoints, mirror::ImageMetaInfo*);

private:
	class Impl;
	Impl* impl_;
};

}

#endif // !_FACE_ALIGNER_H_

