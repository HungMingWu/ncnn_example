#ifndef _FACE_ALIGNER_H_
#define _FACE_ALIGNER_H_

#include <vector>
#include <orbwebai/structure.h>

namespace mirror {
class Aligner {
public:
	Aligner();
	~Aligner();

	int AlignFace(const orbwebai::ImageMetaInfo& img_src,
		const std::vector<orbwebai::Point2f>& keypoints, orbwebai::ImageMetaInfo*);

private:
	class Impl;
	Impl* impl_;
};

}

#endif // !_FACE_ALIGNER_H_

