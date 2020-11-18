#pragma once
#include <memory>
#include <orbwebai/structure.h>
namespace orbwebai
{
	namespace face
	{
		class Tracker
		{
			class Impl;
			std::unique_ptr<Impl> impl;
		public:
			Tracker();
			~Tracker();
			std::vector<orbwebai::face::TrackedInfo> Track(const std::vector<orbwebai::face::Info>& curr_faces);
		};
	}
}