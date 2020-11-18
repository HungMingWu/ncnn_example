#include "classifier_engine.h"
#include "classifier/classifier.h"
#include "classifier/mobilenet/mobilenet.h"

namespace mirror {

class ClassifierEngine::Impl {
public:
	Impl() {
		classifier_ = new Mobilenet();
	}
	~Impl() {
		if (classifier_) {
			delete classifier_;
			classifier_ = nullptr;
		}
	}
	int LoadModel(const char* root_path);
	std::vector<orbwebai::classify::Info> Classify(const orbwebai::ImageMetaInfo& img_src);

private:
	Classifier* classifier_;

};

ClassifierEngine::ClassifierEngine() {
	impl_ = new ClassifierEngine::Impl();
}

ClassifierEngine::~ClassifierEngine() {
	if (impl_) {
		delete impl_;
		impl_ = nullptr;
	}
}

int ClassifierEngine::LoadModel(const char * root_path) {
	return impl_->LoadModel(root_path);
}

std::vector<orbwebai::classify::Info> ClassifierEngine::Classify(const orbwebai::ImageMetaInfo& img_src) {
	return impl_->Classify(img_src);
}



int ClassifierEngine::Impl::LoadModel(const char * root_path) {
	return classifier_->LoadModel(root_path);
}

std::vector<orbwebai::classify::Info> ClassifierEngine::Impl::Classify(const orbwebai::ImageMetaInfo& img_src) {
	return classifier_->Classify(img_src);
}

}


