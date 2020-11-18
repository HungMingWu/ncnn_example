﻿#define _CRT_SECURE_NO_WARNINGS
#include <assert.h>
#include "mobilenet.h"
#include <algorithm>
#include <string>
#include <iostream>

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

namespace mirror {
Mobilenet::Mobilenet() {
	mobilenet_ = new ncnn::Net();
	initialized_ = false;
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    mobilenet_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}

Mobilenet::~Mobilenet() {
	if (mobilenet_) {
		mobilenet_->clear();
	}
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
}

int Mobilenet::LoadModel(const char * root_path) {
	std::cout << "start load model." << std::endl;
	std::string param_file = std::string(root_path) + "/mobilenet.param";
	std::string model_file = std::string(root_path) + "/mobilenet.bin";
	if (mobilenet_->load_param(param_file.c_str()) == -1 ||
		mobilenet_->load_model(model_file.c_str()) == -1 ||
		LoadLabels(root_path) != 0) {
		std::cout << "load model or label file failed." << std::endl;
		return 10000;
	}
	initialized_ = true;
	std::cout << "end load model." << std::endl;

	return 0;
}
std::vector<orbwebai::classify::Info> Mobilenet::Classify(const orbwebai::ImageMetaInfo& img_src) {
	std::cout << "start classify." << std::endl;
	assert(initialized_);
	assert(img_src.data);
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB,
		img_src.width, img_src.height, inputSize.width, inputSize.height);
	in.substract_mean_normalize(meanVals, normVals);

	ncnn::Extractor ex = mobilenet_->create_extractor();
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("prob", out);
	
	std::vector<std::pair<float, int>> scores;
	for (int i = 0; i < out.w; ++i) {
		scores.push_back(std::make_pair(out[i], i));
	}

	int topk = 5;
	std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
		std::greater< std::pair<float, int> >());

	std::vector<orbwebai::classify::Info> images;
	for (int i = 0; i < topk; ++i) {
		orbwebai::classify::Info image_info;
		image_info.label_ = labels_[scores[i].second];
		image_info.score_ = scores[i].first;
		images.push_back(image_info);
	}

	std::cout << "end classify." << std::endl;
	return images;
}

int Mobilenet::LoadLabels(const char * root_path) {
	std::string label_file = std::string(root_path) + "/label.txt";
	FILE* fp = fopen(label_file.c_str(), "r");

	while (!feof(fp)) {
		char str[1024];
		if (nullptr == fgets(str, 1024, fp)) continue;
		std::string str_s(str);

		if (str_s.length() > 0) {
			for (int i = 0; i < str_s.length(); i++) {
				if (str_s[i] == ' ') {
					std::string strr = str_s.substr(i, str_s.length() - i - 1);
					labels_.push_back(strr);
					i = str_s.length();
				}
			}
		}
	}
	return 0;
}
}


