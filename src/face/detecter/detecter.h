#ifndef _FACE_DETECTER_H_
#define _FACE_DETECTER_H_

#include <vector>
#include <orbwebai/structure.h>

namespace mirror {
// 抽象类
class Detecter {
public:
	virtual ~Detecter() {};
	virtual int LoadModel(const char* root_path) = 0;
	virtual std::vector<orbwebai::face::Info> DetectFace(const orbwebai::ImageMetaInfo& img_src) = 0;

};

}

#endif // !_FACE_DETECTER_H_

