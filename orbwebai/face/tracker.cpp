#include <iostream>
#include <queue>
#include <common/common.h>
#include <orbwebai/face/tracker.h>

using namespace orbwebai::face;
class Tracker::Impl {
    std::vector<TrackedInfo> pre_tracked_faces_;
public:
    std::vector<TrackedInfo> Track(const std::vector<Info>& curr_faces) {
        constexpr float minScore_ = 0.3f;
        constexpr float maxScore_ = 0.5f;
        int num_faces = static_cast<int>(curr_faces.size());

        std::deque<TrackedInfo> scored_tracked_faces(pre_tracked_faces_.begin(), pre_tracked_faces_.end());
        std::vector<TrackedInfo> curr_tracked_faces;
        for (int i = 0; i < num_faces; ++i) {
            auto& face = curr_faces.at(i);
            for (auto scored_tracked_face : scored_tracked_faces) {
                scored_tracked_face.iou_score_ = ComputeIOU(scored_tracked_face.face_info_.location_, face.location_);
            }
            if (scored_tracked_faces.size() > 0) {
                std::partial_sort(scored_tracked_faces.begin(),
                    scored_tracked_faces.begin() + 1,
                    scored_tracked_faces.end(),
                    [](const orbwebai::face::TrackedInfo& a, const orbwebai::face::TrackedInfo& b) {
                        return a.iou_score_ > b.iou_score_;
                    });
            }
            if (!scored_tracked_faces.empty() && scored_tracked_faces.front().iou_score_ > minScore_) {
                auto matched_face = scored_tracked_faces.front();
                scored_tracked_faces.pop_front();
                auto &tracked_face = matched_face;
                if (matched_face.iou_score_ < maxScore_) {
                    tracked_face.face_info_.location_.x = (tracked_face.face_info_.location_.x + face.location_.x) / 2;
                    tracked_face.face_info_.location_.y = (tracked_face.face_info_.location_.y + face.location_.y) / 2;
                    tracked_face.face_info_.location_.width = (tracked_face.face_info_.location_.width + face.location_.width) / 2;
                    tracked_face.face_info_.location_.height = (tracked_face.face_info_.location_.height + face.location_.height) / 2;
                }
                else {
                    tracked_face.face_info_ = face;
                }
                curr_tracked_faces.push_back(tracked_face);
            }
            else {
                orbwebai::face::TrackedInfo tracked_face;
                tracked_face.face_info_ = face;
                curr_tracked_faces.push_back(tracked_face);
            }
        }

        pre_tracked_faces_ = curr_tracked_faces;
        return curr_tracked_faces;
    }
};

Tracker::Tracker() : impl(new Tracker::Impl)
{
}

Tracker::~Tracker() = default;

std::vector<TrackedInfo> Tracker::Track(const std::vector<Info>& curr_faces) 
{
    return impl->Track(curr_faces);
}