#ifndef _FACE_CENTERFACE_H_
#define _FACE_CENTERFACE_H_

#include "../detecter.h"
#include <vector>
#include "ncnn/net.h"

namespace mirror {
class CenterFace : public Detecter {
public:
    CenterFace();
    ~CenterFace();
	int LoadModel(const char* root_path);
    std::vector<FaceInfo> DetectFace(const mirror::ImageMetaInfo& img_src) override;

private:
    ncnn::Net* centernet_ = nullptr;
    const float scoreThreshold_ = 0.5f;
    const float nmsThreshold_ = 0.5f;
    bool initialized_;
};

}

#endif // !_FACE_CENTERFACE_H_
