#pragma once
#include <string>
#include <orbwebai/structure.h>
namespace orbwebai
{
	namespace face
	{
		class Detector {
			void* impl;
		public:
			Detector();
			~Detector();
			int LoadModel(const char* root_path);
			std::vector<orbwebai::face::Info> DetectFace(const orbwebai::ImageMetaInfo& img_src);
		};
	}
}
