#include "mappoint.h"

namespace odom
{
  MapPoint::MapPoint() : n_visible_(0) {}

  MapPoint::MapPoint(const cv::Point3d p, const cv::KeyPoint l_kp, const cv::KeyPoint r_kp, const cv::Mat l_desc, const cv::Mat r_desc) : n_visible_(1)
  {
    world_pos_ = p;
    l_kp_ = l_kp;
    r_kp_ = r_kp;
    l_desc.copyTo(l_desc_);
    r_desc.copyTo(r_desc_);
  }

} //namespace odom