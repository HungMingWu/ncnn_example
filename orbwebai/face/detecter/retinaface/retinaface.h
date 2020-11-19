#ifndef _RETINAFACE_H_
#define _RETINAFACE_H_

#include <orbwebai/structure.h>
#include "ncnn/net.h"

namespace orbwebai {
	namespace face
	{
		using ANCHORS = std::vector<orbwebai::Rect>;
		class DetecterBackend final {
		public:
			DetecterBackend();
			~DetecterBackend();
			int LoadModel(const char* root_path);
			std::vector<orbwebai::face::Info> DetectFace(const orbwebai::ImageMetaInfo& img_src);

		private:
			ncnn::Net retina_net_;
			std::vector<ANCHORS> anchors_generated_;
			bool initialized_;
			const int RPNs_[3] = { 32, 16, 8 };
			const orbwebai::Size inputSize_ = { 300, 300 };
			const float iouThreshold_ = 0.4f;
			const float scoreThreshold_ = 0.8f;
		};
	}
}

#endif // !_RETINAFACE_H_

