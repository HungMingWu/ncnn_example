#include <assert.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <orbwebai/structure_operator.h>
#include <common/common.h>

#include "insightface.h"

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

namespace {
	void EnlargeRect(const float& scale, orbwebai::Rect* rect) {
		float offset_x = (scale - 1.f) / 2.f * rect->width;
		float offset_y = (scale - 1.f) / 2.f * rect->height;
		rect->x -= offset_x;
		rect->y -= offset_y;
		rect->width = scale * rect->width;
		rect->height = scale * rect->height;
	}

	void RectifyRect(orbwebai::Rect* rect) 
	{
		int max_side = std::max<int>(rect->width, rect->height);
		int offset_x = (max_side - rect->width) / 2;
		int offset_y = (max_side - rect->height) / 2;

		rect->x -= offset_x;
		rect->y -= offset_y;
		rect->width = max_side;
		rect->height = max_side;
	}
}

using namespace orbwebai::face;
LandmarkerBackend::LandmarkerBackend() {
	initialized = false;
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    insightface_landmarker_net_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}

LandmarkerBackend::~LandmarkerBackend() {
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
}

int LandmarkerBackend::LoadModel(const char * root_path) {
	std::string fl_param = std::string(root_path) + "/2d106.param";
	std::string fl_bin = std::string(root_path) + "/2d106.bin";
	if (insightface_landmarker_net_.load_param(fl_param.c_str()) == -1 ||
		insightface_landmarker_net_.load_model(fl_bin.c_str()) == -1) {
		std::cout << "load face landmark model failed." << std::endl;
		return 10000;
	}
	initialized = true;
	return 0;
}

std::vector<orbwebai::Point2f> LandmarkerBackend::ExtractKeypoints(const orbwebai::ImageMetaInfo& img_src,
	const orbwebai::Rect & face) {
	std::cout << "start extract keypoints." << std::endl;
	assert(initialized);
	assert(img_src.data);
	std::vector<orbwebai::Point2f> keypoints;
	// 1 enlarge the face rect
	orbwebai::Rect face_enlarged = face;
	const float enlarge_scale = 1.5f;
	EnlargeRect(enlarge_scale, &face_enlarged);

	// 2 square the rect
	RectifyRect(&face_enlarged);
	face_enlarged = face_enlarged & orbwebai::Rect(0, 0, img_src.width, img_src.height);

	// 3 crop the face
	std::vector<uint8_t> img_face = CopyImageFromRange(img_src, face_enlarged);

	// 4 do inference
	ncnn::Extractor ex = insightface_landmarker_net_.create_extractor();
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data(),
		ncnn::Mat::PIXEL_BGR2RGB, face_enlarged.width, face_enlarged.height, 192, 192);
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("fc1", out);

	for (int i = 0; i < 106; ++i) {
		float x = (out[2 * i] + 1.0f) * face_enlarged.width / 2 + face_enlarged.x;
		float y = (out[2 * i + 1] + 1.0f) * face_enlarged.height / 2 + face_enlarged.y;
		keypoints.emplace_back(x, y);
	}

	std::cout << "end extract keypoints." << std::endl;
	return keypoints;
}
