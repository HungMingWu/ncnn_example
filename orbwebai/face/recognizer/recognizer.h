#ifndef _FACE_RECOGNIZER_H_
#define _FACE_RECOGNIZER_H_

#include <vector>
#include <orbwebai/structure.h>

namespace mirror {
class Recognizer {
public:
	virtual ~Recognizer() {};
	virtual int LoadModel(const char* root_path) = 0;
	virtual std::vector<float> ExtractFeature(const orbwebai::ImageMetaInfo& img_face) = 0;

};

}

#endif // !_FACE_RECOGNIZER_H_

