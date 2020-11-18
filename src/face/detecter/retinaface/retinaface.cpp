#include "retinaface.h"
#include <iostream>
#include <assert.h>

#if MIRROR_VULKAN
#include "gpu.h"
#endif // MIRROR_VULKAN

namespace mirror {
RetinaFace::RetinaFace() :
	initialized_(false) {
#if MIRROR_VULKAN
	ncnn::create_gpu_instance();	
    retina_net_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN

}

RetinaFace::~RetinaFace() {
#if MIRROR_VULKAN
	ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN	
}

int RetinaFace::LoadModel(const char * root_path) {
	std::string fd_param = std::string(root_path) + "/fd.param";
	std::string fd_bin = std::string(root_path) + "/fd.bin";
	if (retina_net_.load_param(fd_param.c_str()) == -1 ||
		retina_net_.load_model(fd_bin.c_str()) == -1) {
		std::cout << "load face detect model failed." << std::endl;
		return 10000;
	}

	// generate anchors
	for (int i = 0; i < 3; ++i) {
		if (0 == i) {
			anchors_generated_.push_back(GenerateAnchors(16, { 1.0f }, { 32, 16 }));
		}
		else if (1 == i) {
			anchors_generated_.push_back(GenerateAnchors(16, { 1.0f }, { 8, 4 }));
		}
		else {
			anchors_generated_.push_back(GenerateAnchors(16, { 1.0f }, { 2, 1 }));
		}
	}
	initialized_ = true;

	return 0;
}

std::vector<FaceInfo> RetinaFace::DetectFace(const mirror::ImageMetaInfo& img_src) {
	std::cout << "start face detect." << std::endl;
	assert(initialized_);
	assert(img_src.data);
	int img_width = img_src.width;
	int img_height = img_src.height;
	float factor_x = static_cast<float>(img_width) / inputSize_.width;
	float factor_y = static_cast<float>(img_height) / inputSize_.height;
	ncnn::Extractor ex = retina_net_.create_extractor();
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data,
		ncnn::Mat::PIXEL_BGR2RGB, img_width, img_height, inputSize_.width, inputSize_.height);
	ex.input("data", in);
	
	std::vector<FaceInfo> faces_tmp;
	for (int i = 0; i < 3; ++i) {
		std::string class_layer_name = "face_rpn_cls_prob_reshape_stride" + std::to_string(RPNs_[i]);
		std::string bbox_layer_name = "face_rpn_bbox_pred_stride" + std::to_string(RPNs_[i]);
		std::string landmark_layer_name = "face_rpn_landmark_pred_stride" + std::to_string(RPNs_[i]);

		ncnn::Mat class_mat, bbox_mat, landmark_mat;
		ex.extract(class_layer_name.c_str(), class_mat);
		ex.extract(bbox_layer_name.c_str(), bbox_mat);
		ex.extract(landmark_layer_name.c_str(), landmark_mat);

		ANCHORS anchors = anchors_generated_.at(i);
		int width = class_mat.w;
		int height = class_mat.h;
		int anchor_num = static_cast<int>(anchors.size());
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				int index = h * width + w;
				for (int a = 0; a < anchor_num; ++a) {
					float score = class_mat.channel(anchor_num + a)[index];
					if (score < scoreThreshold_) {
						continue;
					}
					// 1.获取anchor生成的box
					mirror::Rect box(w * RPNs_[i] + anchors[a].x,
						h * RPNs_[i] + anchors[a].y,
						anchors[a].width,
						anchors[a].height);

					// 2.解析出偏移量
					float delta_x = bbox_mat.channel(a * 4 + 0)[index];
					float delta_y = bbox_mat.channel(a * 4 + 1)[index];
					float delta_w = bbox_mat.channel(a * 4 + 2)[index];
					float delta_h = bbox_mat.channel(a * 4 + 3)[index];

					// 3.计算anchor box的中心
					mirror::Point2f center(box.x + box.width * 0.5f,
						box.y + box.height * 0.5f);
					
					// 4.计算框的实际中心（anchor的中心+偏移量）
					center.x = center.x + delta_x * box.width;
					center.y = center.y + delta_y * box.height;

					// 5.计算出实际的宽和高
					float curr_width = std::exp(delta_w) * (box.width + 1);
					float curr_height = std::exp(delta_h) * (box.height + 1);

					// 6.获取实际的矩形位置
					mirror::Rect curr_box(center.x - curr_width * 0.5f,
						center.y - curr_height * 0.5f, curr_width, 	curr_height);
					curr_box.x = std::max<int>(curr_box.x * factor_x, 0);
					curr_box.y = std::max<int>(curr_box.y * factor_y, 0);
					curr_box.width = std::min<int>(img_width - curr_box.x, curr_box.width * factor_x);
					curr_box.height = std::min<int>(img_height - curr_box.y, curr_box.height * factor_y);
					
					FaceInfo face_info;

					int offset_index = landmark_mat.c / anchor_num;
					for (int k = 0; k < 5; ++k) {
						float x = landmark_mat.channel(a * offset_index + 2 * k)[index] * box.width + center.x;
						float y = landmark_mat.channel(a * offset_index + 2 * k + 1)[index] * box.height + center.y;
						face_info.keypoints_[k] = std::min<float>(std::max<float>(x * factor_x, 0.0f), img_width - 1);
						face_info.keypoints_[k + 5] = std::min<float>(std::max<float>(y * factor_y, 0.0f), img_height - 1);
					}

					face_info.score_ = score;
					face_info.location_ = curr_box;
					faces_tmp.push_back(face_info);
				}
			}
		}
	}
	
	std::vector<FaceInfo> faces = NMS(faces_tmp, iouThreshold_);
	std::cout << faces.size() << " faces detected." << std::endl;

	std::cout << "end face detect." << std::endl;
	return faces;
}

}
