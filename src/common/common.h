#ifndef _COMMON_H_
#define _COMMON_H_

#include <vector>
#include <string>
#include "opencv2/core.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace mirror {

#define kFaceFeatureDim 128
#define kFaceNameDim 256
const int threads_num = 2;

template<typename _Tp>
struct Rect_
{
    Rect_() : x(0), y(0), width(0), height(0) {}
    Rect_(_Tp _x, _Tp _y, _Tp _w, _Tp _h) : x(_x), y(_y), width(_w), height(_h) {}

    _Tp x;
    _Tp y;
    _Tp width;
    _Tp height;
};

using Rect = Rect_<int>;

template<typename _Tp>
struct Point_
{
    Point_() : x(0), y(0) {}
    Point_(_Tp _x, _Tp _y) : x(_x), y(_y) {}

    _Tp x;
    _Tp y;
};
using Point2f = Point_<float>;
using Point2i = Point_<int>;
using Point = Point2i;

struct ImageInfo {
    std::string label_;
    float score_;
};

struct ObjectInfo {
	cv::Rect location_;
	float score_;
	std::string name_;
};

struct FaceInfo {
	cv::Rect location_;
	float score_;
	float keypoints_[10];
    bool mask_;
};

struct TrackedFaceInfo {
	FaceInfo face_info_;
	float iou_score_;
};

struct QueryResult {
    std::string name_;
    float sim_;
};

std::vector<mirror::Rect> RatioAnchors(const mirror::Rect & anchor,
	const std::vector<float>& ratios);

std::vector<mirror::Rect> ScaleAnchors(const std::vector<mirror::Rect>& ratio_anchors,
	const std::vector<float>& scales);

std::vector<mirror::Rect> GenerateAnchors(const int & base_size,
	const std::vector<float>& ratios, const std::vector<float> scales);

float InterRectArea(const cv::Rect & a,
	const cv::Rect & b);

float ComputeIOU(const cv::Rect & rect1,
	const cv::Rect & rect2,
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

float CalculateSimilarity(const std::vector<float>&feature1, const std::vector<float>& feature2);
void EnlargeRect(const float& scale, cv::Rect* rect);
void RectifyRect(cv::Rect* rect);

}

#endif // !_COMMON_H_


