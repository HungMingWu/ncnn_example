#include "common.h"
#include <algorithm>
#include <iostream>

namespace mirror {
std::vector<mirror::Rect> RatioAnchors(const mirror::Rect & anchor,
	const std::vector<float>& ratios) {
	std::vector<mirror::Rect> anchors;
	mirror::Point center(anchor.x + (anchor.width - 1) * 0.5f,
		anchor.y + (anchor.height - 1) * 0.5f);
	float anchor_size = anchor.width * anchor.height;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads_num)
#endif
	for (int i = 0; i < static_cast<int>(ratios.size()); ++i) {
		float ratio = ratios.at(i);
		float anchor_size_ratio = anchor_size / ratio;
		float curr_anchor_width = std::sqrt(anchor_size_ratio);
		float curr_anchor_height = curr_anchor_width * ratio;
		float curr_x = center.x - (curr_anchor_width - 1)* 0.5f;
		float curr_y = center.y - (curr_anchor_height - 1)* 0.5f;

		anchors.emplace_back(curr_x, curr_y,
                        curr_anchor_width - 1, curr_anchor_height - 1);

	}
	return anchors;
}

std::vector<mirror::Rect> ScaleAnchors(const std::vector<mirror::Rect>& ratio_anchors,
	const std::vector<float>& scales) {
	std::vector<mirror::Rect> anchors;
	for (const auto &anchor : ratio_anchors) {
		mirror::Point2f center(anchor.x + anchor.width * 0.5f,
			anchor.y + anchor.height * 0.5f);
		for (const auto scale : scales) {
			const float curr_width = scale * (anchor.width + 1);
			const float curr_height = scale * (anchor.height + 1);
			const float curr_x = center.x - curr_width * 0.5f;
			const float curr_y = center.y - curr_height * 0.5f;
			anchors.emplace_back(curr_x, curr_y,
                                curr_width, curr_height);
		}
	}

	return anchors;
}

std::vector<mirror::Rect> GenerateAnchors(const int & base_size,
	const std::vector<float>& ratios, 
	const std::vector<float> scales) {
	mirror::Rect anchor(0, 0, base_size, base_size);
	std::vector<mirror::Rect> ratio_anchors = RatioAnchors(anchor, ratios);
	return ScaleAnchors(ratio_anchors, scales);
}

float InterRectArea(const mirror::Rect & a, const mirror::Rect & b) {
	mirror::Point left_top(std::max(a.x, b.x), std::max(a.y, b.y));
	mirror::Point right_bottom(std::min(a.br().x, b.br().x), std::min(a.br().y, b.br().y));
	mirror::Point diff = right_bottom - left_top;
	return (std::max(diff.x + 1, 0) * std::max(diff.y + 1, 0));
}

float ComputeIOU(const mirror::Rect & rect1,
	const mirror::Rect & rect2, const std::string& type) {

	float inter_area = InterRectArea(rect1, rect2);
	if (type == "UNION") {
		return inter_area / (rect1.area() + rect2.area() - inter_area);
	}
	else {
		return inter_area / std::min(rect1.area(), rect2.area());
	}
}

float CalculateSimilarity(const std::vector<float>&feature1, const std::vector<float>& feature2) {
	if (feature1.size() != feature2.size()) {
		std::cout << "feature size not match." << std::endl;
		return 10003;
	}
	float inner_product = 0.0f;
	float feature_norm1 = 0.0f;
	float feature_norm2 = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads_num)
#endif
	for(int i = 0; i < kFaceFeatureDim; ++i) {
		inner_product += feature1[i] * feature2[i];
		feature_norm1 += feature1[i] * feature1[i];
		feature_norm2 += feature2[i] * feature2[i];
	}
	return inner_product / sqrt(feature_norm1) / sqrt(feature_norm2);
}

void EnlargeRect(const float& scale, mirror::Rect* rect) {
        float offset_x = (scale - 1.f) / 2.f * rect->width;
    float offset_y = (scale - 1.f) / 2.f * rect->height;
    rect->x -= offset_x;
    rect->y -= offset_y;
    rect->width = scale * rect->width;
    rect->height = scale * rect->height;
}

void RectifyRect(mirror::Rect* rect) {
        int max_side = std::max(rect->width, rect->height);
        int offset_x = (max_side - rect->width) / 2;
        int offset_y = (max_side - rect->height) / 2;

        rect->x -= offset_x;
        rect->y -= offset_y;
        rect->width = max_side;
        rect->height = max_side;
}

std::vector<uint8_t> CopyImageFromRange(const mirror::ImageMetaInfo& img_src, const mirror::Rect& face)
{
	std::vector<uint8_t> result;
	int channels = img_src.channels;
	for (int y = face.y; y < face.y + face.height; y++)
		for (int x = face.x; x < face.x + face.width; x++)
			for (int c = 0; c < img_src.channels; c++)
				result.push_back(img_src.data[img_src.width * y * channels + x * channels + c]);
	return result;
}

}
