#ifndef _COMMON_H_
#define _COMMON_H_

#include <algorithm>
#include <vector>
#include <string>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace mirror {

#define kFaceFeatureDim 128
#define kFaceNameDim 256
const int threads_num = 2;

struct Size
{
    Size() : width(0), height(0) {}
    Size(int _w, int _h) : width(_w), height(_h) {}

    int width;
    int height;
};

template<typename _Tp>
struct Point_
{
    Point_() : x(0), y(0) {}
    Point_(_Tp _x, _Tp _y) : x(_x), y(_y) {}

    _Tp x;
    _Tp y;
};

template<typename _Tp> static inline
Point_<_Tp> operator - (const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return Point_<_Tp>(a.x - b.x, a.y - b.y);
}

using Point2f = Point_<float>;
using Point2i = Point_<int>;
using Point = Point2i;

template<typename _Tp>
struct Rect_
{
    Rect_() : x(0), y(0), width(0), height(0) {}
    Rect_(_Tp _x, _Tp _y, _Tp _w, _Tp _h) : x(_x), y(_y), width(_w), height(_h) {}

    _Tp x;
    _Tp y;
    _Tp width;
    _Tp height;

    Point_<_Tp> br() const
    {
        return Point_<_Tp>(x + width, y + height);
    }

    _Tp area() const
    {
        return width * height;
    }
};

template<typename _Tp> static inline
Rect_<_Tp>& operator &= (Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    _Tp x1 = std::max(a.x, b.x);
    _Tp y1 = std::max(a.y, b.y);
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    if (a.width <= 0 || a.height <= 0)
        a = Rect_< _Tp>();
    return a;
}

template<typename _Tp> static inline
Rect_<_Tp> operator & (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c &= b;
}

using Rect = Rect_<int>;

struct ImageInfo {
    std::string label_;
    float score_;
};

struct ObjectInfo {
    mirror::Rect location_;
	float score_;
	std::string name_;
};

struct FaceInfo {
	mirror::Rect location_;
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

struct ImageMetaInfo {
    int width;
    int height;
    int channels;
    unsigned char* data;
};

std::vector<mirror::Rect> RatioAnchors(const mirror::Rect & anchor,
	const std::vector<float>& ratios);

std::vector<mirror::Rect> ScaleAnchors(const std::vector<mirror::Rect>& ratio_anchors,
	const std::vector<float>& scales);

std::vector<mirror::Rect> GenerateAnchors(const int & base_size,
	const std::vector<float>& ratios, const std::vector<float> scales);

float InterRectArea(const mirror::Rect & a,
	const mirror::Rect & b);

float ComputeIOU(const mirror::Rect & rect1,
	const mirror::Rect & rect2,
	const std::string& type = "UNION");

std::vector<uint8_t> CopyImageFromRange(const mirror::ImageMetaInfo& img_src, const mirror::Rect& face);

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
void EnlargeRect(const float& scale, mirror::Rect* rect);
void RectifyRect(mirror::Rect* rect);

}

#endif // !_COMMON_H_



