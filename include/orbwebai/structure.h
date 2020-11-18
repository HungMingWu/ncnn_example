#pragma once
#include <string>
#include <vector>

namespace orbwebai 
{
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

        Point_<_Tp> br() const;

        _Tp area() const;
    };

    using Rect = Rect_<int>;

    struct ImageMetaInfo {
        int width;
        int height;
        int channels;
        unsigned char* data;
    };

    namespace face
    {
        struct Info 
        {
            orbwebai::Rect location_;
            float score_;
            float keypoints_[10];
            bool mask_;
        };

        struct TrackedInfo
        {
            Info face_info_;
            float iou_score_;
        };
    }

    namespace object
    {
        struct Info 
        {
            Rect location_;
            float score_;
            std::string name_;
        };
    }

    namespace classify
    {
        struct Info 
        {
            std::string label_;
            float score_;
        };
    }

    namespace query
    {
        struct Result 
        {
            std::string name_;
            float sim_;
        };
    }

    float CalculateSimilarity(const std::vector<float>& feature1, const std::vector<float>& feature2);
}