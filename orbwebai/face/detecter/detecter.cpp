#include <memory>
#include <orbwebai/face/detecter.h>

// Use for detecter_
#include "retinaface/retinaface.h"
#include "mtcnn/mtcnn.h"
#include "anticonv/anticonv.h"
#include "centerface/centerface.h"

using namespace orbwebai::face;
class Detector::Impl {
    using FaceType = RetinaFace;
    std::unique_ptr<IDetecter> detecter_;
public:
    Impl() {
        // detecter_factory_ = new AnticonvFactory();

        detecter_.reset(new FaceType());
    }

    ~Impl() {
    }

    int LoadModel(const char* root_path) {
        return detecter_->LoadModel(root_path);
    }

    std::vector<orbwebai::face::Info> DetectFace(const orbwebai::ImageMetaInfo& img_src) {
        return detecter_->DetectFace(img_src);
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

std::vector<orbwebai::face::Info> Detector::DetectFace(const orbwebai::ImageMetaInfo& img_src)
{
    return  impl->DetectFace(img_src);
}