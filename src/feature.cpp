#include "feature.h"

namespace odom
{
  Feature::Feature() : newest_frame_uid_(0) {}

  Feature::Feature(const uint feat_uid, const uint frame_uid, const cv::Point3d world_pos, const cv::KeyPoint l_kp, const cv::KeyPoint r_kp, const cv::Mat l_desc, const cv::Mat r_desc) : newest_frame_uid_(0)
  {
    addView(feat_uid, frame_uid, world_pos, l_kp, r_kp, l_desc, r_desc);
  }

  void Feature::addView(const uint feat_uid, const uint frame_uid, const cv::Point3d world_pos, const cv::KeyPoint l_kp, const cv::KeyPoint r_kp, const cv::Mat l_desc, const cv::Mat r_desc)
  {
    // Store the newest frame uid
    if (frame_uid > newest_frame_uid_)
      newest_frame_uid_ = frame_uid;

    // Store feature properties
    uids_.push_back(feat_uid);
    world_pos_.push_back(world_pos);
    l_kp_.push_back(l_kp);
    r_kp_.push_back(r_kp);
    l_desc_.push_back(l_desc);
    r_desc_.push_back(r_desc);
  }

  void Feature::addView(const uint feat_uid, const uint frame_uid, const cv::Point3d world_pos, const cv::KeyPoint l_kp, const cv::KeyPoint r_kp)
  {
    // Store the newest frame uid
    if (frame_uid > newest_frame_uid_)
      newest_frame_uid_ = frame_uid;

    // Store feature properties
    uids_.push_back(feat_uid);
    world_pos_.push_back(world_pos);
    l_kp_.push_back(l_kp);
    r_kp_.push_back(r_kp);
  }

} //namespace odom