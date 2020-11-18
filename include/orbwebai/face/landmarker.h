#pragma once
#include <memory>
#include <orbwebai/structure.h>
namespace orbwebai
{
	namespace face
	{
		class Landmarker {
			class Impl;
			std::unique_ptr<Impl> impl;
		public:
			Landmarker();
			~Landmarker();
			int LoadModel(const char* root_path);
			std::vector<orbwebai::Point2f> ExtractKeypoints(const orbwebai::ImageMetaInfo& img_src,
				const orbwebai::Rect& face);
		};
	}
}