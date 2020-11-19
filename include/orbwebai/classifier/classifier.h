#pragma once
#include <string>
#include <memory>
#include <orbwebai/structure.h>
namespace orbwebai
{
	namespace classify
	{
		class Classifier {
			class Impl;
			std::unique_ptr<Impl> impl;
		public:
			Classifier();
			~Classifier();
			int LoadModel(const char* root_path);
			std::vector<orbwebai::classify::Info> Classify(const orbwebai::ImageMetaInfo& img_src);
		};
	}
}
