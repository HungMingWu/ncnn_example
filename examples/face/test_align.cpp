#define FACE_EXPORTS
#include "opencv2/opencv.hpp"
#include <orbwebai/face/detecter.h>
#include <orbwebai/face/landmarker.h>
#include <orbwebai/face/compute.h>
#include "../image_helper.h"

int TestAlignFace(int argc, char* argv[]) {
	const char* img_file = "../../../..//data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_file);
	const char* root_path = "../../../..//data/models";

	double start = static_cast<double>(cv::getTickCount());
	
	auto faceDetector = make_unique<orbwebai::face::Detector>();
	faceDetector->LoadModel(root_path);
	auto faces = faceDetector->DetectFace(toImageInfo(img_src));
	auto faceLandmarker = make_unique<orbwebai::face::Landmarker>();
	faceLandmarker->LoadModel(root_path);
	constexpr int alignWidth = 112;
	constexpr int alignHeight = 112;
	constexpr int alignChannel = 3;
	orbwebai::ImageMetaInfo face_aligned;
	face_aligned.width = alignWidth;
	face_aligned.height = alignHeight;
	face_aligned.channels = alignChannel;
	face_aligned.data = new unsigned char[alignWidth * alignHeight * alignChannel];
	for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
		auto face = faces.at(i).location_;
		auto keypoints = faceLandmarker->ExtractKeypoints(toImageInfo(img_src), face);

		orbwebai::face::AlignFace(toImageInfo(img_src), keypoints, &face_aligned);
		cv::Mat extract_image(112, 112, CV_8UC3, face_aligned.data);
		std::string name = std::to_string(i) + ".jpg";
		cv::imwrite(name.c_str(), extract_image);
		for (int j = 0; j < static_cast<int>(keypoints.size()); ++j) {
			cv::circle(img_src, toPoint(keypoints[j]), 1, cv::Scalar(0, 0, 255), 1);
		}
		cv::rectangle(img_src, toRect(face), cv::Scalar(0, 255, 0), 2);
	}
	cv::imshow("result", img_src);
	cv::waitKey(0);

	delete[] face_aligned.data;
	
	return 0;
}

int main(int argc, char* argv[]) {
	// return TestRecognize(argc, argv);
	return TestAlignFace(argc, argv);
	// return TestTrack(argc, argv);
	// return TestMask(argc, argv);
}
