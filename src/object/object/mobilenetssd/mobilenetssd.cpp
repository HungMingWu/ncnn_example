#include "mobilenetssd.h"

#include <iostream>
#include <string>
#include <assert.h>
#include <common/common.h>

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

namespace mirror {
MobilenetSSD::MobilenetSSD() :
	mobilenetssd_(new ncnn::Net()),
	initialized_(false) {
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    mobilenetssd_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN

}

MobilenetSSD::~MobilenetSSD() {
	mobilenetssd_->clear();
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN
}

int MobilenetSSD::LoadModel(const char * root_path) {
	std::cout << "start load model." << std::endl;
	std::string obj_param = std::string(root_path) + "/mobilenetssd.param";
	std::string obj_bin = std::string(root_path) + "/mobilenetssd.bin";
	if (mobilenetssd_->load_param(obj_param.c_str()) == -1 ||
		mobilenetssd_->load_model(obj_bin.c_str()) == -1) {
		std::cout << "load ssd model failed." << std::endl;
		return 10000;
	}

	initialized_ = true;
	std::cout << "end load model." << std::endl;

	return 0;
}

std::vector<orbwebai::object::Info> MobilenetSSD::DetectObject(const orbwebai::ImageMetaInfo& img_src) {
	std::cout << "start object detect." << std::endl;
	assert(initialized_);
	assert(img_src.data);
	int width = img_src.width;
	int height = img_src.height;

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data,
		ncnn::Mat::PIXEL_BGR, img_src.width, img_src.height, 300, 300);
	in.substract_mean_normalize(meanVals, normVals);

	ncnn::Extractor ex = mobilenetssd_->create_extractor();
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("detection_out", out);

	std::vector<orbwebai::object::Info> objects_tmp;
	for (int i = 0; i < out.h; i++) {
		const float* values = out.row(i);
		orbwebai::object::Info object;
		object.name_ = class_names[int(values[0])];
		object.score_ = values[1];
		object.location_.x = values[2] * width;
		object.location_.y = values[3] * height;
		object.location_.width = values[4] * width - object.location_.x;
		object.location_.height = values[5] * height - object.location_.y;

		// filter the result
		if (object.score_ < scoreThreshold_) {
			continue;
		}
		objects_tmp.push_back(object);
	}
	auto objects = NMS(objects_tmp, nmsThreshold_);
	std::cout << "objects number: " << objects.size() << std::endl;
	std::cout << "end object detect." << std::endl;
	return objects;
}

}
