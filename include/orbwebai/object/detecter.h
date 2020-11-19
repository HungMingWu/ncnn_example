#pragma once
#include <string>
#include <memory>
#include <orbwebai/structure.h>
namespace orbwebai
{
	namespace object
	{
		class Detector {
			class Impl;
			std::unique_ptr<Impl> impl;
		public:
			Detector();
			~Detector();
			int LoadModel(const char* root_path);
			std::vector<orbwebai::object::Info> DetectObject(const orbwebai::ImageMetaInfo& img_src);
		};
	}
}
