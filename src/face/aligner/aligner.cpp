#include <iostream>
#include <Eigen/Geometry>
#include <orbwebai/structure.h>
#include "aligner.h"

namespace mirror {
class Aligner::Impl {
public:
	int AlignFace(const orbwebai::ImageMetaInfo& img_src,
		const std::vector<orbwebai::Point2f>& keypoints, orbwebai::ImageMetaInfo*);

private:
	float points_dst[5][2] = {
		{ 30.2946f + 8.0f, 51.6963f },
		{ 65.5318f + 8.0f, 51.5014f },
		{ 48.0252f + 8.0f, 71.7366f },
		{ 33.5493f + 8.0f, 92.3655f },
		{ 62.7299f + 8.0f, 92.2041f }
	};
};


Aligner::Aligner() {
	impl_ = new Impl();
}

Aligner::~Aligner() {
	if (impl_) {
		delete impl_;
	}
}

int Aligner::AlignFace(const orbwebai::ImageMetaInfo& img_src,
	const std::vector<orbwebai::Point2f>& keypoints, orbwebai::ImageMetaInfo*p) {
	return impl_->AlignFace(img_src, keypoints, p);
}

int Aligner::Impl::AlignFace(const orbwebai::ImageMetaInfo& img_src,
	const std::vector<orbwebai::Point2f>& keypoints, orbwebai::ImageMetaInfo*p) {
	std::cout << "start align face." << std::endl;
	assert(img_src.data);
	assert(keypoints.size() > 0);

	float points_src[5][2] = {
		{keypoints[104].x, keypoints[104].y},
		{keypoints[105].x, keypoints[105].y},
		{keypoints[46].x,  keypoints[46].y },
		{keypoints[84].x,  keypoints[84].y },
		{keypoints[90].x, keypoints[90].y}
	};

	Eigen::Matrix<float, 2, 5> umeyamaSrc;
	umeyamaSrc(0, 0) = points_src[0][0];
	umeyamaSrc(1, 0) = points_src[0][1];

	umeyamaSrc(0, 1) = points_src[1][0];
	umeyamaSrc(1, 1) = points_src[1][1];

	umeyamaSrc(0, 2) = points_src[2][0];
	umeyamaSrc(1, 2) = points_src[2][1];

	umeyamaSrc(0, 3) = points_src[3][0];
	umeyamaSrc(1, 3) = points_src[3][1];

	umeyamaSrc(0, 4) = points_src[4][0];
	umeyamaSrc(1, 4) = points_src[4][1];

	Eigen::Matrix<float, 2, 5> umeyamaDest;

	umeyamaDest(0, 0) = points_dst[0][0];
	umeyamaDest(1, 0) = points_dst[0][1];

	umeyamaDest(0, 1) = points_dst[1][0];
	umeyamaDest(1, 1) = points_dst[1][1];

	umeyamaDest(0, 2) = points_dst[2][0];
	umeyamaDest(1, 2) = points_dst[2][1];

	umeyamaDest(0, 3) = points_dst[3][0];
	umeyamaDest(1, 3) = points_dst[3][1];

	umeyamaDest(0, 4) = points_dst[4][0];
	umeyamaDest(1, 4) = points_dst[4][1];

	auto inverse_trans = Eigen::umeyama(umeyamaDest, umeyamaSrc);

	int channels = p->channels;
	for (int dsty = 0; dsty < p->height; dsty++)
		for (int dstx = 0; dstx < p->width; dstx++) {
			int srcx = static_cast<int>(dstx * inverse_trans(0, 0) + dsty * inverse_trans(0, 1) + inverse_trans(0, 2));
			int srcy = static_cast<int>(dstx * inverse_trans(1, 0) + dsty * inverse_trans(1, 1) + inverse_trans(1, 2));
			if (srcx >= 0 && srcx < img_src.width && srcy >= 0 && srcy < img_src.height)
			{
				auto* src_ptr = &img_src.data[img_src.width * srcy * channels + srcx * channels];
				auto* dst_ptr = &p->data[p->width * dsty * channels + dstx * channels];
				for (int c = 0; c < channels; c++)
					dst_ptr[c] = src_ptr[c];
			}
		}

	std::cout << "end align face." << std::endl;
	return 0;
}

}
