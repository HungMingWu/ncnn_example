#include <memory>
#include <orbwebai/face/recognizer.h>

#include "mobilefacenet/mobilefacenet.h"

using namespace orbwebai::face;
class Recognizer::Impl {
    std::unique_ptr<mirror::Recognizer> recognizer_;
public:
    Impl() {
        recognizer_.reset(new mirror::Mobilefacenet());
    }

    ~Impl() {
    }

    int LoadModel(const char* root_path) {
        return recognizer_->LoadModel(root_path);
    }

    std::vector<float> ExtractFeature(const orbwebai::ImageMetaInfo& img_face)
    {
        return recognizer_->ExtractFeature(img_face);
    }
};

Recognizer::Recognizer() : impl(new Recognizer::Impl)
{
}

Recognizer::~Recognizer() = default;

int Recognizer::LoadModel(const char* root_path)
{
    return impl->LoadModel(root_path);
}

std::vector<float> Recognizer::ExtractFeature(const orbwebai::ImageMetaInfo& img_face)
{
    return impl->ExtractFeature(img_face);
}