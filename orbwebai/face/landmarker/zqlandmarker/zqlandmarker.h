#ifndef _FACE_ZQLANDMARKER_H_
#define _FACE_ZQLANDMARKER_H_

#include "../landmarker.h"
#include "ncnn/net.h"

namespace mirror {
class ZQLandmarker : public Landmarker {
public:
	ZQLandmarker();
	~ZQLandmarker();

	int LoadModel(const char* root_path);
	std::vector<orbwebai::Point2f> ExtractKeypoints(const orbwebai::ImageMetaInfo& img_src,
		const orbwebai::Rect& face) override;

private:
	ncnn::Net* zq_landmarker_net_;
	const float meanVals[3] = { 127.5f, 127.5f, 127.5f };
	const float normVals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
	bool initialized;
};

}

#endif // !_FACE_ZQLANDMARKER_H_

