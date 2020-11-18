#include "insightface.h"
#include <iostream>
#include <string>
#include "../../common/common.h"
#include <assert.h>

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

namespace mirror {
InsightfaceLandmarker::InsightfaceLandmarker() {
	initialized = false;
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    insightface_landmarker_net_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}

InsightfaceLandmarker::~InsightfaceLandmarker() {
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
}

int InsightfaceLandmarker::LoadModel(const char * root_path) {
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

std::vector<mirror::Point2f> InsightfaceLandmarker::ExtractKeypoints(const mirror::ImageMetaInfo& img_src,
	const mirror::Rect & face) {
	std::cout << "start extract keypoints." << std::endl;
	assert(initialized);
	assert(img_src.data);
	std::vector<mirror::Point2f> keypoints;
	// 1 enlarge the face rect
	mirror::Rect face_enlarged = face;
	const float enlarge_scale = 1.5f;
	EnlargeRect(enlarge_scale, &face_enlarged);

	// 2 square the rect
	RectifyRect(&face_enlarged);
	face_enlarged = face_enlarged & mirror::Rect(0, 0, img_src.width, img_src.height);

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

}
