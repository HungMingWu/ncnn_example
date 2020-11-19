#include <memory>
#include <orbwebai/face/detecter.h>

// Use for detecter_
#if defined(USE_RETIMAFCE)
#include "retinaface/retinaface.h"
#elif defined(USE_MTCNN)
#include "mtcnn/mtcnn.h"
#elif defined(USE_CENTERFACE)
#include "centerface/centerface.h"
#else
#include "anticonv/anticonv.h"
#endif

using namespace orbwebai::face;
Detector::Detector()
{
    impl = new DetecterBackend();
}

Detector::~Detector()
{
    delete static_cast<DetecterBackend*>(impl);
}

int Detector::LoadModel(const char* root_path)
{
    return static_cast<DetecterBackend*>(impl)->LoadModel(root_path);
}

std::vector<orbwebai::face::Info> Detector::DetectFace(const orbwebai::ImageMetaInfo& img_src)
{
    return static_cast<DetecterBackend*>(impl)->DetectFace(img_src);
}