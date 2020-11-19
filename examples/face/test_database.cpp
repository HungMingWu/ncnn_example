#define FACE_EXPORTS
#include "opencv2/opencv.hpp"
#include <orbwebai/face/detecter.h>
#include <orbwebai/face/landmarker.h>
#include <orbwebai/face/recognizer.h>
#include <orbwebai/face/tracker.h>
#include <orbwebai/face/database.h>
#include <orbwebai/face/compute.h>
#include "../image_helper.h"

int main(int argc, char* argv[]) {
	const char* img_path = "../../data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_path);
	if (img_src.empty()) {
		std::cout << "load image failed." << std::endl;
		return 10001;
	}

	const char* root_path = "../../data/models";
	auto faceDatabase = make_unique<orbwebai::face::Database>(root_path);
	faceDatabase->Load();
	auto faceDetector = make_unique<orbwebai::face::Detector>();
	faceDetector->LoadModel(root_path);
	auto faces = faceDetector->DetectFace(toImageInfo(img_src));
	auto faceRecognizer = make_unique<orbwebai::face::Recognizer>();
	faceRecognizer->LoadModel(root_path);
	int faces_num = static_cast<int>(faces.size());
	std::cout << "faces number: " << faces_num << std::endl;
	for (int i = 0; i < faces_num; ++i) {
		auto face = faces.at(i).location_;
		cv::rectangle(img_src, toRect(face), cv::Scalar(0, 255, 0), 2);
		cv::Mat img_cpy = img_src(toRect(face)).clone();
		std::vector<float> feat = faceRecognizer->ExtractFeature(toImageInfo(img_cpy));

#if 1
		faceDatabase->Insert(feat, "face" + std::to_string(i));
#endif

#if 0
		orbwebai::query::Result query_result;
		faceDatabase->QueryTop(feat, &query_result);
		std::cout << i << "-th face is: " << query_result.name_ <<
			" similarity is: " << query_result.sim_ << std::endl;
#endif

	}
	faceDatabase->Save();
	cv::imwrite("../../data/images/result.jpg", img_src);

	return 0;
}
