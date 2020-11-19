#ifndef _FACE_MTCNN_H_
#define _FACE_MTCNN_H_

#include <orbwebai/structure.h>
#include <vector>
#include "ncnn/net.h"

namespace orbwebai {
	namespace face
	{
		class DetecterBackend final {
		public:
			DetecterBackend();
			~DetecterBackend();
			int LoadModel(const char* root_path);
			std::vector<orbwebai::face::Info> DetectFace(const orbwebai::ImageMetaInfo& img_src);

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
			std::vector<orbwebai::face::Info> PDetect(const ncnn::Mat& img_in);
			std::vector<orbwebai::face::Info> RDetect(const ncnn::Mat& img_in,
				const std::vector<orbwebai::face::Info>& first_bboxes);
			std::vector<orbwebai::face::Info> ODetect(const ncnn::Mat& img_in,
				const std::vector<orbwebai::face::Info>& second_bboxes);

			int Refine(std::vector<orbwebai::face::Info>& bboxes, const orbwebai::Size max_size);
		};
	}
}


#endif // !_FACE_MTCNN_H_

