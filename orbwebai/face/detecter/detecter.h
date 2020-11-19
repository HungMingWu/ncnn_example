#ifndef _FACE_DETECTER_H_
#define _FACE_DETECTER_H_

#include <vector>
#include <orbwebai/structure.h>

namespace orbwebai {
	namespace face
	{
		class IDetecter {
		public:
			virtual ~IDetecter() {};
			virtual int LoadModel(const char* root_path) = 0;
			virtual std::vector<orbwebai::face::Info> DetectFace(const orbwebai::ImageMetaInfo& img_src) = 0;
		};
	}
}

#endif // !_FACE_DETECTER_H_

