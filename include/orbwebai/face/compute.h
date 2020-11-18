#pragma once
#include <orbwebai/structure.h>
namespace orbwebai
{
	namespace face
	{
		float CalculateSimilarity(const std::vector<float>& feature1, const std::vector<float>& feature2);
		void AlignFace(const orbwebai::ImageMetaInfo& img_src, const std::vector<orbwebai::Point2f>& keypoints,
			orbwebai::ImageMetaInfo* output);
	}
}