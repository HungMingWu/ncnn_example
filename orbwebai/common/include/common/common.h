#ifndef _COMMON_H_
#define _COMMON_H_

#include <algorithm>
#include <vector>
#include <string>
#include <orbwebai/structure.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace orbwebai
{
    static constexpr size_t kFaceFeatureDim = 128;
    static constexpr size_t threads_num = 2;
    static constexpr size_t kFaceNameDim = 256;

    std::vector<orbwebai::Rect> GenerateAnchors(const int& base_size,
        const std::vector<float>& ratios, const std::vector<float> scales);

    float ComputeIOU(const orbwebai::Rect& rect1,
        const orbwebai::Rect& rect2,
        const std::string& type = "UNION");

    template <typename T>
    std::vector<T> NMS(const std::vector<T>& inputs,
        const float& threshold, const std::string& type = "UNION") {
        std::vector<T> result;
        if (inputs.size() == 0)
            return {};

        std::vector<T> inputs_tmp;
        inputs_tmp.assign(inputs.begin(), inputs.end());
        std::sort(inputs_tmp.begin(), inputs_tmp.end(),
            [](const T& a, const T& b) {
                return a.score_ > b.score_;
            });

        std::vector<int> indexes(inputs_tmp.size());

        for (int i = 0; i < indexes.size(); i++) {
            indexes[i] = i;
        }

        while (indexes.size() > 0) {
            int good_idx = indexes[0];
            result.push_back(inputs_tmp[good_idx]);
            std::vector<int> tmp_indexes = indexes;
            indexes.clear();
            for (int i = 1; i < tmp_indexes.size(); i++) {
                int tmp_i = tmp_indexes[i];
                float iou = ComputeIOU(inputs_tmp[good_idx].location_, inputs_tmp[tmp_i].location_, type);
                if (iou <= threshold) {
                    indexes.push_back(tmp_i);
                }
            }
        }
        return result;
    }

    std::vector<uint8_t> CopyImageFromRange(const orbwebai::ImageMetaInfo& img_src, const orbwebai::Rect& face);
}

#endif // !_COMMON_H_



