#ifndef _FACE_INSIGHTFACE_LANDMARKER_H_
#define _FACE_INSIGHTFACE_LANDMARKER_H_

#include "../landmarker.h"
#include "ncnn/net.h"

namespace mirror {
class InsightfaceLandmarker : public Landmarker {
public:
	InsightfaceLandmarker();
	~InsightfaceLandmarker();

	int LoadModel(const char* root_path);
	std::vector<mirror::Point2f> ExtractKeypoints(const mirror::ImageMetaInfo& img_src,
		const mirror::Rect& face) override;

private:
	ncnn::Net insightface_landmarker_net_;
	bool initialized;
};

}

#endif // !_FACE_INSIGHTFACE_LANDMARKER_H_

