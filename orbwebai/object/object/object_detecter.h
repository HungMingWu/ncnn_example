#ifndef _OBJECT_DETECTER_H_
#define _OBJECT_DETECTER_H_

#include <vector>
#include <orbwebai/structure.h>

namespace orbwebai {
	namespace object
	{
		class IDetecter {
		public:
			virtual ~IDetecter() {}
			virtual int LoadModel(const char* root_path) = 0;
			virtual std::vector<orbwebai::object::Info> DetectObject(const orbwebai::ImageMetaInfo& img_src) = 0;
		};
	}

}



#endif // !_OBJECT_DETECT_H_

