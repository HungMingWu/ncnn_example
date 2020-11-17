#ifndef _OBJECT_DETECTER_H_
#define _OBJECT_DETECTER_H_

#include <vector>
#include "opencv2/core.hpp"
#include "../object_engine.h"

namespace mirror {
class ObjectDetecter {
public:
	virtual ~ObjectDetecter() {}
	virtual int LoadModel(const char* root_path) = 0;
	virtual std::vector<ObjectInfo> DetectObject(const mirror::ImageMetaInfo& img_src) = 0;
};

}



#endif // !_OBJECT_DETECT_H_

