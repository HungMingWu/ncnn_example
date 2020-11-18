#ifndef _FACE_MTCNN_H_
#define _FACE_MTCNN_H_

#include "../detecter.h"
#include <vector>
#include "ncnn/net.h"

namespace mirror {
class Mtcnn : public Detecter {
public:
	Mtcnn();
	~Mtcnn();
	int LoadModel(const char* root_path);
	std::vector<FaceInfo> DetectFace(const mirror::ImageMetaInfo& img_src) override;

private:
	ncnn::Net* pnet_ = nullptr;
	ncnn::Net* rnet_ = nullptr;
	ncnn::Net* onet_ = nullptr;
	int pnet_size_;
	int min_face_size_;
	float scale_factor_;
	bool initialized_;
	const float meanVals[3] = { 127.5f, 127.5f, 127.5f };
	const float normVals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
	const float nms_threshold_[3] = { 0.5f, 0.7f, 0.7f };
	const float threshold_[3] = { 0.8f, 0.8f, 0.6f };

private:
	std::vector<FaceInfo> PDetect(const ncnn::Mat& img_in);
	std::vector<FaceInfo> RDetect(const ncnn::Mat& img_in, const std::vector<FaceInfo>& first_bboxes);
	std::vector<FaceInfo> ODetect(const ncnn::Mat& img_in,
		const std::vector<FaceInfo>& second_bboxes);

	int Refine(std::vector<FaceInfo>& bboxes, const mirror::Size max_size);
};

}


#endif // !_FACE_MTCNN_H_

