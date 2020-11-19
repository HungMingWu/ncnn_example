#ifndef _FACE_MOBILEFACENET_H_
#define _FACE_MOBILEFACENET_H_

#include "../recognizer.h"
#include <vector>
#include "ncnn/net.h"

namespace orbwebai {
	namespace face {
		class Mobilefacenet : public IRecognizer {
		public:
			Mobilefacenet();
			~Mobilefacenet();

			int LoadModel(const char* root_path);
			std::vector<float> ExtractFeature(const orbwebai::ImageMetaInfo& img_face) override;

		private:
			ncnn::Net mobileface_net_;
			bool initialized_;
		};
	}
}

#endif // !_FACE_MOBILEFACENET_H_

