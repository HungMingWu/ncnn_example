#include <orbwebai/classifier/classifier.h>
#include "classifier/mobilenet/mobilenet.h"

using namespace orbwebai::classify;
class Classifier::Impl {
public:
	Impl() {
		classifier_.reset(new mirror::Mobilenet());
	}
	~Impl() = default;
	int LoadModel(const char* root_path)
	{
		return classifier_->LoadModel(root_path);
	}
	std::vector<orbwebai::classify::Info> Classify(const orbwebai::ImageMetaInfo& img_src)
	{
		return classifier_->Classify(img_src);
	}

private:
	std::unique_ptr<mirror::Classifier> classifier_;

};

Classifier::Classifier() : impl(new Classifier::Impl)
{
}

Classifier::~Classifier() = default;

int Classifier::LoadModel(const char * root_path) {
	return impl->LoadModel(root_path);
}

std::vector<orbwebai::classify::Info> Classifier::Classify(const orbwebai::ImageMetaInfo& img_src) 
{
	return impl->Classify(img_src);
}