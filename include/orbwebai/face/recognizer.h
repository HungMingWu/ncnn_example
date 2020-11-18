#pragma once
#include <memory>
#include <orbwebai/structure.h>
namespace orbwebai
{
	namespace face
	{
		class Recognizer
		{
			class Impl;
			std::unique_ptr<Impl> impl;
		public:
			Recognizer();
			~Recognizer();
			int LoadModel(const char* root_path);
			std::vector<float> ExtractFeature(const orbwebai::ImageMetaInfo& img_face);
		};
	}
}