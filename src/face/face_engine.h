#include <vector>
#include <orbwebai/structure.h>

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
	FACE_API std::vector<orbwebai::face::Info> DetectFace(const orbwebai::ImageMetaInfo& img_src);
	FACE_API std::vector<orbwebai::face::TrackedInfo> Track(const std::vector<orbwebai::face::Info>& curr_faces);
	FACE_API std::vector<orbwebai::Point2f> ExtractKeypoints(const orbwebai::ImageMetaInfo& img_src,
		const orbwebai::Rect& face);
	FACE_API std::vector<float> ExtractFeature(const orbwebai::ImageMetaInfo& img_face);
	FACE_API int AlignFace(const orbwebai::ImageMetaInfo& img_src, const std::vector<orbwebai::Point2f>& keypoints,
		orbwebai::ImageMetaInfo*);

	// database operation
    FACE_API int Insert(const std::vector<float>& feat, const std::string& name);
	FACE_API int Delete(const std::string& name);
	FACE_API int64_t QueryTop(const std::vector<float>& feat, orbwebai::query::Result*query_result = nullptr);
    FACE_API int Save();
    FACE_API int Load();

private:
	class Impl;
	Impl* impl_;

};

}
