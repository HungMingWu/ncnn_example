#ifndef _FACE_CENTERFACE_H_
#define _FACE_CENTERFACE_H_

#include "../detecter.h"
#include <vector>
#include "ncnn/net.h"

namespace orbwebai {
    namespace face
    {
        class CenterFace : public IDetecter {
        public:
            CenterFace();
            ~CenterFace();
            int LoadModel(const char* root_path);
            std::vector<orbwebai::face::Info> DetectFace(const orbwebai::ImageMetaInfo& img_src) override;

        private:
            ncnn::Net* centernet_ = nullptr;
            const float scoreThreshold_ = 0.5f;
            const float nmsThreshold_ = 0.5f;
            bool initialized_;
        };
    }
}

#endif // !_FACE_CENTERFACE_H_
