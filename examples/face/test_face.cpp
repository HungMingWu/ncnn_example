#define FACE_EXPORTS
#include "opencv2/opencv.hpp"
#include <orbwebai/face/detecter.h>
#include <orbwebai/face/landmarker.h>
#include <orbwebai/face/recognizer.h>
#include <orbwebai/face/tracker.h>
#include <orbwebai/face/database.h>
#include <orbwebai/face/compute.h>
#include "../image_helper.h"

int TestLandmark(int argc, char* argv[]) {
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

int TestDatabase(int argc, char* argv[]) {
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
	// return TestLandmark(argc, argv);
	// return TestRecognize(argc, argv);
	return TestAlignFace(argc, argv);
	// return TestTrack(argc, argv);
	// return TestDatabase(argc, argv);
	// return TestMask(argc, argv);
}
