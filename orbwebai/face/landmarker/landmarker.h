#ifndef _FACE_LANDMARKER_H_
#define _FACE_LANDMARKER_H_

#include <vector>
#include <orbwebai/structure.h>

namespace mirror {
// 抽象类
class Landmarker {
public:
	virtual ~Landmarker() {};
	virtual int LoadModel(const char* root_path) = 0;
	virtual std::vector<orbwebai::Point2f> ExtractKeypoints(const orbwebai::ImageMetaInfo& img_src,
		const orbwebai::Rect& face) = 0;
};

}

#endif // !_FACE_LANDMARKER_H_

