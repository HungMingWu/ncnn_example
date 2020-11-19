#include <assert.h>
#include <iostream>
#include <string>
#include <common/common.h>
#include "zqlandmarker.h"

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

using namespace orbwebai::face;
LandmarkerBackend::LandmarkerBackend() {
	zq_landmarker_net_ = new ncnn::Net();
	initialized = false;
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    zq_landmarker_net_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}

LandmarkerBackend::~LandmarkerBackend() {
	zq_landmarker_net_->clear();
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
}

int LandmarkerBackend::LoadModel(const char * root_path) {
	std::string fl_param = std::string(root_path) + "/fl.param";
	std::string fl_bin = std::string(root_path) + "/fl.bin";
	if (zq_landmarker_net_->load_param(fl_param.c_str()) == -1 ||
		zq_landmarker_net_->load_model(fl_bin.c_str()) == -1) {
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
	auto crop_image = CopyImageFromRange(img_src, face);
	ncnn::Extractor ex = zq_landmarker_net_->create_extractor();
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(crop_image.data(),
		ncnn::Mat::PIXEL_BGR, face.width, face.height, 112, 112);
	in.substract_mean_normalize(meanVals, normVals);
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("bn6_3", out);

	for (int i = 0; i < 106; ++i) {
		float x = abs(out[2 * i] * face.width) + face.x;
		float y = abs(out[2 * i + 1] * face.height) + face.y;
		keypoints.emplace_back(x, y);
	}

	std::cout << "end extract keypoints." << std::endl;
	return keypoints;
}