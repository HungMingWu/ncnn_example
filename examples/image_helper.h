#pragma once
#include <opencv2/opencv.hpp>
#include <orbwebai/structure.h>

inline orbwebai::ImageMetaInfo toImageInfo(const cv::Mat& img)
{
	return orbwebai::ImageMetaInfo{ img.cols, img.rows, img.channels(), img.data };
}

inline cv::Rect toRect(orbwebai::Rect rect)
{
	return cv::Rect(rect.x, rect.y, rect.width, rect.height);
}

inline cv::Point2f toPoint(orbwebai::Point2f point)
{
	return cv::Point2d(point.x, point.y);
}

template <typename T, typename ...Ts>
std::unique_ptr<T> make_unique(Ts&& ...args)
{
	return std::unique_ptr<T>(new T(std::forward<Ts>(args)...));
}
