#ifndef _FACE_ZQLANDMARKER_H_
#define _FACE_ZQLANDMARKER_H_

#include "ncnn/net.h"

namespace orbwebai {
	namespace face
	{
		class LandmarkerBackend final {
		public:
			LandmarkerBackend();
			~LandmarkerBackend();

			int LoadModel(const char* root_path);
			std::vector<orbwebai::Point2f> ExtractKeypoints(const orbwebai::ImageMetaInfo& img_src,
				const orbwebai::Rect& face);

		private:
			ncnn::Net* zq_landmarker_net_;
			const float meanVals[3] = { 127.5f, 127.5f, 127.5f };
			const float normVals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
			bool initialized;
		};
	}
}

#endif // !_FACE_ZQLANDMARKER_H_

