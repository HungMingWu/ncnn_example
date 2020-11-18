#include "face_engine.h"
#include <iostream>
#include <memory>

#include "detecter/detecter.h"
#include "tracker/tracker.h"
#include "landmarker/landmarker.h"
#include "aligner/aligner.h"
#include "recognizer/recognizer.h"
#include "database/face_database.h"

// Use for detecter_
#include "retinaface/retinaface.h"
#include "mtcnn/mtcnn.h"
#include "anticonv/anticonv.h"
#include "centerface/centerface.h"

// Use for landmarker
#include "zqlandmarker/zqlandmarker.h"
#include "insightface/insightface.h"

#include "mobilefacenet/mobilefacenet.h"

namespace mirror {
class FaceEngine::Impl {
    using FaceType = RetinaFace;
    using LandmarkType = ZQLandmarker;
public:
    Impl() {
        // detecter_factory_ = new AnticonvFactory();
        
        detecter_.reset(new FaceType());
        landmarker_.reset(new LandmarkType());
        recognizer_.reset(new Mobilefacenet());

		tracker_ = new Tracker();
        aligner_ = new Aligner();
        database_ = new FaceDatabase();
        initialized_ = false;
    }

    ~Impl() {
		if (tracker_) {
			delete tracker_;
			tracker_ = nullptr;
		}

        if (database_) {
            delete database_;
            database_ = nullptr;
        }
    }

    int LoadModel(const char* root_path) {
        if (detecter_->LoadModel(root_path) != 0) {
            std::cout << "load face detecter failed." << std::endl;
            return 10000;
        }

        if (landmarker_->LoadModel(root_path) != 0) {
            std::cout << "load face landmarker failed." << std::endl;
            return 10000;
        }

        if (recognizer_->LoadModel(root_path) != 0) {
            std::cout << "load face recognizer failed." << std::endl;
            return 10000;
        }

        db_name_ = std::string(root_path);
        initialized_ = true;

        return 0;
    }
    inline std::vector<TrackedFaceInfo> Track(const std::vector<FaceInfo>& curr_faces) {
        return tracker_->Track(curr_faces);
    }
    inline std::vector<FaceInfo> DetectFace(const mirror::ImageMetaInfo& img_src) {
        return detecter_->DetectFace(img_src);
    }
    inline std::vector<mirror::Point2f> ExtractKeypoints(const mirror::ImageMetaInfo& img_src,
		const mirror::Rect& face) {
        return landmarker_->ExtractKeypoints(img_src, face);
    }
    inline int AlignFace(const mirror::ImageMetaInfo& img_src, const std::vector<mirror::Point2f>& keypoints,
        mirror::ImageMetaInfo* p) {
        return aligner_->AlignFace(img_src, keypoints, p);
    }
    inline std::vector<float> ExtractFeature(const mirror::ImageMetaInfo& img_face) {
        return recognizer_->ExtractFeature(img_face);
    }

    inline int Insert(const std::vector<float>& feat, const std::string& name) {
        return database_->Insert(feat, name);
    }
    inline int Delete(const std::string& name) {
        return database_->Delete(name);
    }
	inline int64_t QueryTop(const std::vector<float>& feat, QueryResult *query_result = nullptr) {
        return database_->QueryTop(feat, query_result);
    }
    inline int Save() {
        return  database_->Save(db_name_.c_str());
    }
    inline int Load() {
        return database_->Load(db_name_.c_str());
    }

private:
    bool initialized_;
    std::string db_name_;
    Aligner* aligner_ = nullptr;
    std::unique_ptr<Detecter> detecter_;
	Tracker* tracker_ = nullptr;
    std::unique_ptr<Landmarker> landmarker_;
    std::unique_ptr<Recognizer> recognizer_;
    FaceDatabase* database_ = nullptr;
};

FaceEngine::FaceEngine() {
    impl_ = new FaceEngine::Impl();
}

FaceEngine::~FaceEngine() {
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }
}

int FaceEngine::LoadModel(const char* root_path) {
    return impl_->LoadModel(root_path);
}

std::vector<TrackedFaceInfo> FaceEngine::Track(const std::vector<FaceInfo>& curr_faces) {
	return impl_->Track(curr_faces);
}

std::vector<FaceInfo> FaceEngine::DetectFace(const mirror::ImageMetaInfo& img_src) {
    return impl_->DetectFace(img_src);
}

std::vector<mirror::Point2f> FaceEngine::ExtractKeypoints(const mirror::ImageMetaInfo& img_src,
	const mirror::Rect& face) {
    return impl_->ExtractKeypoints(img_src, face);
}

int FaceEngine::AlignFace(const mirror::ImageMetaInfo& img_src, const std::vector<mirror::Point2f>& keypoints,
    mirror::ImageMetaInfo* p) {
    return impl_->AlignFace(img_src, keypoints, p);
}

std::vector<float> FaceEngine::ExtractFeature(const mirror::ImageMetaInfo& img_face) {
    return impl_->ExtractFeature(img_face);
}

int FaceEngine::Insert(const std::vector<float>& feat, const std::string& name) {
    return impl_->Insert(feat, name);
}

int FaceEngine::Delete(const std::string& name) {
    return impl_->Delete(name);
}

int64_t FaceEngine::QueryTop(const std::vector<float>& feat,
    QueryResult* query_result) {
    return impl_->QueryTop(feat, query_result);
}

int FaceEngine::Save() {
    return impl_->Save();
}

int FaceEngine::Load() {
    return impl_->Load();
}

}
