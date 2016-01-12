#include "frame.h"

namespace odom
{
  Frame::Frame() {}

  Frame::Frame(Featools* feat, const cv::Mat l_img, const cv::Mat r_img, const uint frame_uid, uint& feat_uid) : featools_(feat)
  {
    // Copy
    l_img.copyTo(l_img_);
    r_img.copyTo(r_img_);

    // Extract keypoints
    vector<cv::KeyPoint> l_kp, r_kp;
    cv::ORB orb(1500, 1.2, 8, 10, 0, 2, 0, 10);
    orb(l_img_, cv::noArray(), l_kp, cv::noArray(), false);
    orb(r_img_, cv::noArray(), r_kp, cv::noArray(), false);

    // Extract descriptors
    cv::Mat l_desc, r_desc;
    cv::Ptr<cv::DescriptorExtractor> cv_extractor;
    cv_extractor = cv::DescriptorExtractor::create("ORB");
    cv_extractor->compute(l_img_, l_kp, l_desc);
    cv_extractor->compute(r_img_, r_kp, r_desc);

    if (l_kp.empty() || r_kp.empty()) return;

    // Undistort
    featools_->undistortKeyPoints(l_kp, l_ukp_);
    featools_->undistortKeyPoints(r_kp, r_ukp_);

    // Left/right matching
    vector<cv::DMatch> matches;
    featools_->ratioMatching(l_desc, r_desc, 0.8, matches);

    // Filter
    featools_->stereoMatchingFilter(l_ukp_, r_ukp_, matches, matches_filtered_);

    // Fill the features vector
    for (uint i=0; i<matches_filtered_.size(); i++)
    {
      cv::KeyPoint l_kp_1     = l_ukp_[matches_filtered_[i].queryIdx];
      cv::KeyPoint r_kp_1     = r_ukp_[matches_filtered_[i].trainIdx];
      cv::Mat l_desc_1        = l_desc.row(matches_filtered_[i].queryIdx);
      cv::Mat r_desc_1        = r_desc.row(matches_filtered_[i].trainIdx);
      cv::Point3d world_point = computeWorldPoint(l_kp_1, r_kp_1);

      Feature* f = new Feature(feat_uid, frame_uid, world_point, l_kp_1, r_kp_1, l_desc_1, r_desc_1);
      features_.push_back(f);
      feat_uid++;
    }
  }

  cv::Point3d Frame::computeWorldPoint(cv::KeyPoint l_kp, cv::KeyPoint r_kp)
  {
    // Camera parameters
    double baseline = featools_->getBaseline();
    cv::Mat camera_matrix = featools_->getCameraMatrix();
    ROS_ASSERT(camera_matrix.at<double>(0,0) == camera_matrix.at<double>(1,1));
    const double cx = camera_matrix.at<double>(0,2);
    const double cy = camera_matrix.at<double>(1,2);
    const double f  = camera_matrix.at<double>(0,0);

    double disparity = l_kp.pt.x - r_kp.pt.x;
    double wa = (1.0 / baseline) * disparity;
    double x = (l_kp.pt.x - cx) * (1.0 / wa);
    double y = (l_kp.pt.y - cy) * (1.0 / wa);
    double z = f * (1.0 / wa);
    cv::Point3d p(x, y, z);
    return p;
  }

  cv::Mat Frame::drawStereoMatches()
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
        l_kp.push_back(l_ukp_[matches_filtered_[i].queryIdx]);
        r_kp.push_back(r_ukp_[matches_filtered_[i].trainIdx]);

        matches_filtered_[i].queryIdx = i;
        matches_filtered_[i].trainIdx = i;
      }

      // Draw
      cv::drawMatches(l_img_, l_kp, r_img_, r_kp, matches_filtered_, img_matches);
      return img_matches;
    }
  }

} //namespace odom