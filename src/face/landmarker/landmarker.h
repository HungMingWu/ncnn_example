#ifndef _FACE_LANDMARKER_H_
#define _FACE_LANDMARKER_H_

#include "common.h"

namespace mirror {
// 抽象类
class Landmarker {
public:
	virtual ~Landmarker() {};
	virtual int LoadModel(const char* root_path) = 0;
	virtual std::vector<mirror::Point2f> ExtractKeypoints(const mirror::ImageMetaInfo& img_src,
		const mirror::Rect& face) = 0;
};

}

#endif // !_FACE_LANDMARKER_H_

