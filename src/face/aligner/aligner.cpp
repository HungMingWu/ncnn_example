#include "aligner.h"
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <Eigen/Geometry>
namespace mirror {
class Aligner::Impl {
public:
	int AlignFace(const cv::Mat& img_src, const std::vector<mirror::Point2f>& keypoints, cv::Mat* face_aligned);

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

int Aligner::AlignFace(const cv::Mat & img_src,
	const std::vector<mirror::Point2f>& keypoints, cv::Mat * face_aligned) {
	return impl_->AlignFace(img_src, keypoints, face_aligned);
}

int Aligner::Impl::AlignFace(const cv::Mat & img_src,
	const std::vector<mirror::Point2f>& keypoints, cv::Mat * face_aligned) {
	std::cout << "start align face." << std::endl;
	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}
	if (keypoints.size() == 0) {
		std::cout << "keypoints empty." << std::endl;
		return 10001;
	}

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

	auto trans = Eigen::umeyama(umeyamaSrc, umeyamaDest);
	face_aligned->create(112, 112, CV_32FC3);

	cv::Mat transfer_mat(2, 3, CV_32FC1);
	transfer_mat.at<float>(0, 0) = trans(0, 0);
	transfer_mat.at<float>(0, 1) = trans(0, 1);
	transfer_mat.at<float>(0, 2) = trans(0, 2);
	transfer_mat.at<float>(1, 0) = trans(1, 0);
	transfer_mat.at<float>(1, 1) = trans(1, 1);
	transfer_mat.at<float>(1, 2) = trans(1, 2);
	cv::warpAffine(img_src.clone(), *face_aligned, transfer_mat, cv::Size(112, 112), 1, 0, 0);

	std::cout << "end align face." << std::endl;
	return 0;
}

}
