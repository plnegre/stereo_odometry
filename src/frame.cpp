#include "frame.h"

namespace odom
{

  Frame::Frame() : feat_(new Feature) {}

  Frame::Frame(const cv::Mat l_img, const cv::Mat r_img, const cv::Mat camera_matrix, const cv::Mat dist_coef, const double baseline) : feat_(new Feature)
  {
    // Copy
    l_img.copyTo(l_img_);
    r_img.copyTo(r_img_);
    camera_matrix.copyTo(camera_matrix_);
    dist_coef.copyTo(dist_coef_);
    baseline_ = baseline;

    // Init feature tools
    feat_->setParameters(camera_matrix, dist_coef, baseline);

    // Extract keypoints
    vector<cv::KeyPoint> l_kp, r_kp;
    cv::ORB orb(1500, 1.2, 8, 10, 0, 2, 0, 10);
    orb(l_img_, cv::noArray(), l_kp, cv::noArray(), false);
    orb(r_img_, cv::noArray(), r_kp, cv::noArray(), false);

    // Extract descriptors
    cv::Ptr<cv::DescriptorExtractor> cv_extractor;
    cv_extractor = cv::DescriptorExtractor::create("ORB");
    cv_extractor->compute(l_img_, l_kp, l_desc_);
    cv_extractor->compute(r_img_, r_kp, r_desc_);

    if (l_kp.empty() || r_kp.empty()) return;

    // Undistort
    feat_->undistortKeyPoints(l_kp, l_kp_);
    feat_->undistortKeyPoints(r_kp, r_kp_);

    // Left/right matching
    vector<cv::DMatch> matches;
    feat_->ratioMatching(l_desc_, r_desc_, 0.8, matches);

    // Filter
    feat_->stereoMatchingFilter(l_kp_, r_kp_, matches, matches_filtered_);
  }

  vector<cv::KeyPoint> Frame::getLeftKp()
  {
    vector<cv::KeyPoint> kps;
    for (uint i=0; i<matches_filtered_.size(); i++)
      kps.push_back(l_kp_[matches_filtered_[i].queryIdx]);

    return kps;
  }

  vector<cv::KeyPoint> Frame::getRightKp()
  {
    vector<cv::KeyPoint> kps;
    for (uint i=0; i<matches_filtered_.size(); i++)
      kps.push_back(r_kp_[matches_filtered_[i].trainIdx]);

    return kps;
  }

  cv::Mat Frame::getLeftDesc()
  {
    cv::Mat desc;
    for (uint i=0; i<matches_filtered_.size(); i++)
      desc.push_back(l_desc_.row(matches_filtered_[i].queryIdx));

    return desc;
  }

  cv::Mat Frame::getRightDesc()
  {
    cv::Mat desc;
    for (uint i=0; i<matches_filtered_.size(); i++)
      desc.push_back(r_desc_.row(matches_filtered_[i].trainIdx));

    return desc;
  }

  vector<MapPoint*> Frame::getMapPoints()
  {
    ROS_ASSERT(camera_matrix_.at<double>(0,0) == camera_matrix_.at<double>(1,1));

    // Camera parameters
    const double cx = camera_matrix_.at<double>(0,2);
    const double cy = camera_matrix_.at<double>(1,2);
    const double f = camera_matrix_.at<double>(0,0);

    vector<MapPoint*> mps;
    for (uint i=0; i<matches_filtered_.size(); i++)
    {
      cv::KeyPoint l_kp = l_kp_[matches_filtered_[i].queryIdx];
      cv::KeyPoint r_kp = r_kp_[matches_filtered_[i].trainIdx];

      // Compute 3D
      double disparity = l_kp.pt.x - r_kp.pt.x;
      double wa = (1.0 / baseline_) * disparity;
      double x = (l_kp.pt.x - cx) * (1.0 / wa);
      double y = (l_kp.pt.y - cy) * (1.0 / wa);
      double z = f * (1.0 / wa);
      cv::Point3d p(x, y, z);

      // Extract best descriptor
      cv::Mat l_desc = l_desc_.row(matches_filtered_[i].queryIdx);
      cv::Mat r_desc = r_desc_.row(matches_filtered_[i].trainIdx);

      MapPoint* mp = new MapPoint(p, l_kp, r_kp, l_desc, r_desc);
      mps.push_back(mp);
    }

    return mps;
  }

  cv::Mat Frame::drawMatches()
  {
    cv::Mat img_matches;

    if (matches_filtered_.size() == 0)
    {
      cv::hconcat(l_img_, r_img_, img_matches);
      return img_matches;
    }
    else
    {
      // Draw matches only
      vector<cv::KeyPoint> l_kp, r_kp;
      for (uint i=0; i<matches_filtered_.size(); i++)
      {
        l_kp.push_back(l_kp_[matches_filtered_[i].queryIdx]);
        r_kp.push_back(r_kp_[matches_filtered_[i].trainIdx]);

        matches_filtered_[i].queryIdx = i;
        matches_filtered_[i].trainIdx = i;
      }

      // Draw
      cv::drawMatches(l_img_, l_kp, r_img_, r_kp, matches_filtered_, img_matches);
      return img_matches;
    }


  }

} //namespace odom