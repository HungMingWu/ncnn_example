#include <memory>
#include <orbwebai/object/detecter.h>
#include "object/mobilenetssd/mobilenetssd.h"

using namespace orbwebai::object;
class Detector::Impl {
    std::unique_ptr<mirror::ObjectDetecter> object_detecter_;
public:
    Impl() {
        // detecter_factory_ = new AnticonvFactory();

        object_detecter_.reset(new mirror::MobilenetSSD());
    }

    ~Impl() {
    }

    int LoadModel(const char* root_path) {
        return object_detecter_->LoadModel(root_path);
    }

    std::vector<orbwebai::object::Info> DetectObject(const orbwebai::ImageMetaInfo& img_src) {
        return object_detecter_->DetectObject(img_src);
    }
};

Detector::Detector() : impl(new Detector::Impl)
{
}

Detector::~Detector() = default;

int Detector::LoadModel(const char* root_path)
{
    return impl->LoadModel(root_path);
}

std::vector<orbwebai::object::Info> Detector::DetectObject(const orbwebai::ImageMetaInfo & img_src)
{
    return  impl->DetectObject(img_src);
}