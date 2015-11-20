#include "mappoint.h"

namespace odom
{
  MapPoint::MapPoint() : n_visible_(0) {}

  MapPoint::MapPoint(const cv::Point3d p, const cv::Mat desc, const float kp_resp) : n_visible_(1)
  {
    world_pos_ = p;
    desc.copyTo(desc_);
    kp_resp_ = kp_resp;
  }

} //namespace odom