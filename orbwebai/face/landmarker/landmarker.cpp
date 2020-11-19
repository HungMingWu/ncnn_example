#include <memory>
#include <orbwebai/face/landmarker.h>

// Use for landmarker
#if defined(USE_INSIGHTFACE)
#include "insightface/insightface.h"
#else
#include "zqlandmarker/zqlandmarker.h"
#endif

using namespace orbwebai::face;

Landmarker::Landmarker() : impl(new LandmarkerBackend)
{
}

Landmarker::~Landmarker()
{
    delete static_cast<LandmarkerBackend*>(impl);
}

int Landmarker::LoadModel(const char* root_path)
{
    return static_cast<LandmarkerBackend*>(impl)->LoadModel(root_path);
}
std::vector<orbwebai::Point2f> Landmarker::ExtractKeypoints(const orbwebai::ImageMetaInfo& img_src,
    const orbwebai::Rect& face)
{
    return static_cast<LandmarkerBackend*>(impl)->ExtractKeypoints(img_src, face);
}