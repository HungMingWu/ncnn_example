#define FACE_EXPORTS
#include "opencv2/opencv.hpp"
#include <orbwebai/face/detecter.h>
#include <orbwebai/face/landmarker.h>
#include "../image_helper.h"

int main(int argc, char* argv[]) {
	const char* img_file = "../..//data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_file);
	const char* root_path = "../..//data/models";

	double start = static_cast<double>(cv::getTickCount());

	auto faceDetector = make_unique<orbwebai::face::Detector>();
	auto faceLandmarker = make_unique<orbwebai::face::Landmarker>();
	faceDetector->LoadModel(root_path);
	faceLandmarker->LoadModel(root_path);
	auto faces = faceDetector->DetectFace(toImageInfo(img_src));
	for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
		orbwebai::Rect face = faces.at(i).location_;
		std::vector<orbwebai::Point2f> keypoints = faceLandmarker->ExtractKeypoints(toImageInfo(img_src), face);
		for (int j = 0; j < static_cast<int>(keypoints.size()); ++j) {
			cv::circle(img_src, toPoint(keypoints[j]), 1, cv::Scalar(0, 0, 255), 1);
		}
		cv::rectangle(img_src, toRect(face), cv::Scalar(0, 255, 0), 2);
	}

	double end = static_cast<double>(cv::getTickCount());
	double time_cost = (end - start) / cv::getTickFrequency() * 1000;
	std::cout << "time cost: " << time_cost << "ms" << std::endl;
	cv::imwrite("../../images/result.jpg", img_src);
	cv::imshow("result", img_src);
	cv::waitKey(0);

	return 0;
}
