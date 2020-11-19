#ifndef _FACE_INSIGHTFACE_LANDMARKER_H_
#define _FACE_INSIGHTFACE_LANDMARKER_H_

#include <orbwebai/structure.h>
#include "ncnn/net.h"

namespace orbwebai
{
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
			ncnn::Net insightface_landmarker_net_;
			bool initialized;
		};
	}
}

#endif // !_FACE_INSIGHTFACE_LANDMARKER_H_

