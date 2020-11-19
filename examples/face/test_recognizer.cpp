#define FACE_EXPORTS
#include "opencv2/opencv.hpp"
#include <orbwebai/face/detecter.h>
#include <orbwebai/face/recognizer.h>
#include <orbwebai/face/compute.h>
#include "../image_helper.h"

int TestRecognize(int argc, char* argv[]) {
	const char* img_file = "../../data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_file);
	const char* root_path = "../../data/models";

	double start = static_cast<double>(cv::getTickCount());
	auto faceDetector = make_unique<orbwebai::face::Detector>();
	faceDetector->LoadModel(root_path);
	auto faces = faceDetector->DetectFace(toImageInfo(img_src));
	auto faceRecognizer = make_unique<orbwebai::face::Recognizer>();
	faceRecognizer->LoadModel(root_path);
	cv::Mat face1 = img_src(toRect(faces[0].location_)).clone();
	cv::Mat face2 = img_src(toRect(faces[1].location_)).clone();
	std::vector<float> feature1 = faceRecognizer->ExtractFeature(toImageInfo(face1));
	std::vector<float> feature2 = faceRecognizer->ExtractFeature(toImageInfo(face2));
	float sim = orbwebai::face::CalculateSimilarity(feature1, feature2);

	double end = static_cast<double>(cv::getTickCount());
	double time_cost = (end - start) / cv::getTickFrequency() * 1000;
	std::cout << "time cost: " << time_cost << "ms" << std::endl;

	for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
		orbwebai::Rect face = faces.at(i).location_;
		cv::rectangle(img_src, toRect(face), cv::Scalar(0, 255, 0), 2);
	}
	cv::imwrite("../../data/images/face1.jpg", face1);
	cv::imwrite("../../data/images/face2.jpg", face2);
	cv::imwrite("result.jpg", img_src);
	std::cout << "similarity is: " << sim << std::endl;

	return 0;

}

int TestTrack(int argc, char* argv[]) {
	const char* img_file = "../../data/images/4.jpg";
	cv::Mat img_src = cv::imread(img_file);
	cv::VideoCapture cam(0);
	if (!cam.isOpened()) {
		std::cout << "open camera failed." << std::endl;
		return -1;
	}

	const char* root_path = "../../data/models";

	auto faceDetector = make_unique<orbwebai::face::Detector>();
	faceDetector->LoadModel(root_path);
	auto faceTracker = make_unique<orbwebai::face::Tracker>();
	cv::Mat frame;
	while (true) {
		cam >> frame;
		if (frame.empty()) {
			continue;
		}
		auto curr_faces = faceDetector->DetectFace(toImageInfo(frame));
		auto faces = faceTracker->Track(curr_faces);

		for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
			auto tracked_face_info = faces.at(i);
			cv::rectangle(frame, toRect(tracked_face_info.face_info_.location_), cv::Scalar(0, 255, 0), 2);
		}

		cv::imshow("result", frame);
		if (cv::waitKey(60) == 'q') {
			break;
		}
	}

	return 0;
}

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
	return TestRecognize(argc, argv);
	// return TestTrack(argc, argv);
	// return TestMask(argc, argv);
}
