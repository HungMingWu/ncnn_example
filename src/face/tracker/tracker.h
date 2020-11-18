#ifndef _FACE_TRACKER_H_
#define _FACE_TRACKER_H_

#include <vector>
#include <orbwebai/structure.h>

namespace mirror {
class Tracker {
public:
    Tracker();
    ~Tracker();
    std::vector<orbwebai::face::TrackedInfo> Track(const std::vector<orbwebai::face::Info>& curr_faces);

private:
    std::vector<orbwebai::face::TrackedInfo> pre_tracked_faces_;
    const float minScore_ = 0.3f;
    const float maxScore_ = 0.5f;
};

}

#endif // !_FACE_TRACKER_H_
