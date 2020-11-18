#pragma once
#include <string>
#include <memory>
#include <orbwebai/structure.h>
namespace orbwebai
{
	namespace face
	{
		class Detector {
			class Impl;
			std::unique_ptr<Impl> impl;
		public:
			Detector(const std::string &db_path);
			~Detector();
			int LoadModel(const char* root_path);
			std::vector<orbwebai::face::Info> DetectFace(const orbwebai::ImageMetaInfo& img_src);
		};
	}
}