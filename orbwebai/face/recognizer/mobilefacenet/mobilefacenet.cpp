#include "mobilefacenet.h"
#include <iostream>
#include <string>
#include <assert.h>
#include <common/common.h>

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

using namespace orbwebai::face;

Mobilefacenet::Mobilefacenet() {
	initialized_ = false;
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    mobileface_net_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}

Mobilefacenet::~Mobilefacenet() {
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
}

int Mobilefacenet::LoadModel(const char * root_path) {
	std::string fr_param = std::string(root_path) + "/fr.param";
	std::string fr_bin = std::string(root_path) + "/fr.bin";
	if (mobileface_net_.load_param(fr_param.c_str()) == -1 ||
		mobileface_net_.load_model(fr_bin.c_str()) == -1) {
		std::cout << "load face recognize model failed." << std::endl;
		return 10000;
	}

	initialized_ = true;
	return 0;
}

std::vector<float> Mobilefacenet::ExtractFeature(const orbwebai::ImageMetaInfo& img_face) {
	std::cout << "start extract feature." << std::endl;
	assert(initialized_);
	assert(img_face.data);
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data,
		ncnn::Mat::PIXEL_BGR2RGB, img_face.width, img_face.height, 112, 112);
	ncnn::Extractor ex = mobileface_net_.create_extractor();
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("fc1", out);
	std::vector<float> features(orbwebai::kFaceFeatureDim);
	for (size_t i = 0; i < orbwebai::kFaceFeatureDim; ++i) {
		features[i] = out[i];
	}

	std::cout << "end extract feature." << std::endl;

	return features;
}


