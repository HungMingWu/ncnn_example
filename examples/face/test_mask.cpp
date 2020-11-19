#define FACE_EXPORTS
#include "opencv2/opencv.hpp"
#include <orbwebai/face/detecter.h>
#include <orbwebai/face/recognizer.h>
#include <orbwebai/face/compute.h>
#include "../image_helper.h"

int TestMask(int argc, char* argv[]) {
	const char* img_file = "../../data/images/mask3.jpg";
	cv::Mat img_src = cv::imread(img_file);
	const char* root_path = "../../data/models";

	auto faceDetector = make_unique<orbwebai::face::Detector>();
	faceDetector->LoadModel(root_path);
	double start = static_cast<double>(cv::getTickCount());
	auto faces = faceDetector->DetectFace(toImageInfo(img_src));
	double end = static_cast<double>(cv::getTickCount());
	double time_cost = (end - start) / cv::getTickFrequency() * 1000;
	std::cout << "time cost: " << time_cost << "ms" << std::endl;

	int num_face = static_cast<int>(faces.size());
	for (int i = 0; i < num_face; ++i) {
		if (faces[i].mask_) {
			cv::rectangle(img_src, toRect(faces[i].location_), cv::Scalar(0, 255, 0), 2);
		} else {
			cv::rectangle(img_src, toRect(faces[i].location_), cv::Scalar(0, 0, 255), 2);
		}
	}
	cv::imwrite("../../data/images/mask_result.jpg", img_src);
	cv::imshow("result", img_src);
	cv::waitKey(0);

	return 0;
}


int main(int argc, char* argv[]) {
	return TestMask(argc, argv);
}
