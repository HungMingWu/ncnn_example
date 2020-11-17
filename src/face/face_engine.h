#include <vector>
#include "../common/common.h"
#include "opencv2/core.hpp"

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
    #ifdef FACE_EXPORTS
        #define FACE_API __declspec(dllexport)
    #else
        #define FACE_API __declspec(dllimport)
    #endif
#else
    #define FACE_API __attribute__ ((visibility("default")))
#endif

namespace mirror {
class FaceEngine {
public:
	FACE_API FaceEngine();
	FACE_API ~FaceEngine();
	FACE_API int LoadModel(const char* root_path);
	FACE_API std::vector<FaceInfo> DetectFace(const mirror::ImageMetaInfo& img_src);
	FACE_API std::vector<TrackedFaceInfo> Track(const std::vector<FaceInfo>& curr_faces);
	FACE_API std::vector<mirror::Point2f> ExtractKeypoints(const mirror::ImageMetaInfo& img_src,
		const mirror::Rect& face);
	FACE_API std::vector<float> ExtractFeature(const mirror::ImageMetaInfo& img_face);
	FACE_API int AlignFace(const cv::Mat& img_src, const std::vector<mirror::Point2f>& keypoints, cv::Mat* face_aligned);

	// database operation
    FACE_API int Insert(const std::vector<float>& feat, const std::string& name);
	FACE_API int Delete(const std::string& name);
	FACE_API int64_t QueryTop(const std::vector<float>& feat, QueryResult *query_result = nullptr);
    FACE_API int Save();
    FACE_API int Load();

private:
	class Impl;
	Impl* impl_;

};

}
