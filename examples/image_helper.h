#pragma once
#include <opencv2/opencv.hpp>
#include "common.h"

inline mirror::ImageMetaInfo toImageInfo(const cv::Mat& img)
{
	return mirror::ImageMetaInfo{ img.cols, img.rows, img.channels(), img.data };
}

inline cv::Rect toRect(mirror::Rect rect)
{
	return cv::Rect(rect.x, rect.y, rect.width, rect.height);
}

inline cv::Point2f toPoint(mirror::Point2f point)
{
	return cv::Point2d(point.x, point.y);
}
