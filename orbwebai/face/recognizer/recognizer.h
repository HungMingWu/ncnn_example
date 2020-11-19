#ifndef _FACE_RECOGNIZER_H_
#define _FACE_RECOGNIZER_H_

#include <vector>
#include <orbwebai/structure.h>

namespace orbwebai {
	namespace face
	{
		class IRecognizer {
		public:
			virtual ~IRecognizer() {};
			virtual int LoadModel(const char* root_path) = 0;
			virtual std::vector<float> ExtractFeature(const orbwebai::ImageMetaInfo& img_face) = 0;
		};
	}
}

#endif // !_FACE_RECOGNIZER_H_

