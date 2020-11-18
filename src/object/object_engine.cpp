#include "object_engine.h"
#include <iostream>
#include <string>
#include "object/object_detecter.h"
#include "object/mobilenetssd/mobilenetssd.h"

namespace mirror {
class ObjectEngine::Impl {
public:
	Impl() {
		object_detecter_ = new MobilenetSSD();
		initialized_ = false;
	}
	~Impl() {
	}
	
	inline int LoadModel(const char* root_path) {
		if (object_detecter_->LoadModel(root_path) != 0) {
			return 10000;
		}
		initialized_ = true;
		
		return 0;
	}

	inline std::vector<orbwebai::object::Info>  DetectObject(const orbwebai::ImageMetaInfo& img_src) {
		return object_detecter_->DetectObject(img_src);
	}

private:
	ObjectDetecter* object_detecter_ = nullptr;
	bool initialized_;

};


ObjectEngine::ObjectEngine() {
	impl_ = new Impl();
}

ObjectEngine::~ObjectEngine() {
	if (impl_) {
		delete impl_;
		impl_ = nullptr;
	}
}


int ObjectEngine::LoadModel(const char * root_path) {
	return impl_->LoadModel(root_path);
}

std::vector<orbwebai::object::Info> ObjectEngine::DetectObject(const orbwebai::ImageMetaInfo& img_src) {
	return impl_->DetectObject(img_src);
}

}

