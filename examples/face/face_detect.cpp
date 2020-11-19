#define FACE_EXPORTS
#include "opencv2/opencv.hpp"
#include "../image_helper.h"
#include <orbwebai/face/detecter.h>

int main(int argc, char* argv[]) {
	const char* img_file = "../../../..//data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_file);
	const char* root_path = "../../../..//data/models";

	auto faceDetector = make_unique<orbwebai::face::Detector>();
	faceDetector->LoadModel(root_path);
	double start = static_cast<double>(cv::getTickCount());
	auto faces = faceDetector->DetectFace(toImageInfo(img_src));
	double end = static_cast<double>(cv::getTickCount());
	double time_cost = (end - start) / cv::getTickFrequency() * 1000;
	std::cout << "time cost: " << time_cost << "ms" << std::endl;

	for (const auto& face_info : faces) {
		cv::rectangle(img_src, toRect(face_info.location_), cv::Scalar(0, 255, 0), 2);
		for (int num = 0; num < 5; ++num) {
			cv::Point curr_pt = cv::Point(face_info.keypoints_[num],
				face_info.keypoints_[num + 5]);
			cv::circle(img_src, curr_pt, 2, cv::Scalar(255, 0, 255), 2);
		}
	}
	cv::imwrite("../../../../data/images/retinaface_result.jpg", img_src);
	cv::imshow("result", img_src);
	cv::waitKey(0);

	return 0;
}
