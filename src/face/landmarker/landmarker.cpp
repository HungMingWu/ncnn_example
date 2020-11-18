#include <memory>
#include <detecter.h>
#include <orbwebai/face/landmarker.h>

// Use for landmarker
#include "zqlandmarker/zqlandmarker.h"
#include "insightface/insightface.h"

using namespace orbwebai::face;
class Landmarker::Impl {
    using LandmarkType = mirror::ZQLandmarker;
    std::unique_ptr<mirror::Landmarker> landmarker_;
public:
    Impl() {
        landmarker_.reset(new LandmarkType());
    }

    ~Impl() {
    }

    int LoadModel(const char* root_path) {
        return landmarker_->LoadModel(root_path);
    }

    std::vector<orbwebai::Point2f> ExtractKeypoints(const orbwebai::ImageMetaInfo& img_src,
        const orbwebai::Rect& face)
    {
        return landmarker_->ExtractKeypoints(img_src, face);
    }
};

Landmarker::Landmarker() : impl(new Landmarker::Impl)
{
}

Landmarker::~Landmarker() = default;

int Landmarker::LoadModel(const char* root_path)
{
    return impl->LoadModel(root_path);
}
std::vector<orbwebai::Point2f> Landmarker::ExtractKeypoints(const orbwebai::ImageMetaInfo& img_src,
    const orbwebai::Rect& face)
{
    return impl->ExtractKeypoints(img_src, face);
}