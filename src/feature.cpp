#include "feature.h"

namespace odom
{
  Feature::Feature() : max_disparity_(0.0) {}

  void Feature::setParameters(const cv::Mat camera_matrix, const cv::Mat dist_coef, const double baseline)
  {
    // Copy
    camera_matrix.copyTo(camera_matrix_);
    dist_coef.copyTo(dist_coef_);
    baseline_ = baseline;

    // Minimum depth range
    const double min_depth = 0.5;

    // Minimum and maximum Z
    cv::Point3d p_min(baseline_/2.0, 0.0, 0.7);
    std::vector<cv::Point3d> object_points;
    object_points.push_back(p_min);

    // Rotation matrix
    cv::Mat rvec = cv::Mat::eye(3, 3, cv::DataType<double>::type);
    cv::Mat rvec_rod(3, 1, cv::DataType<double>::type);
    cv::Rodrigues(rvec, rvec_rod);

    // Translation vector
    cv::Mat trans_left = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    cv::Mat trans_right = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    trans_right.at<double>(0) = baseline_;

    vector<cv::Point2d> l_projected_points, r_projected_points;
    cv::projectPoints(object_points, rvec_rod, trans_left, camera_matrix_, dist_coef_, l_projected_points);
    cv::projectPoints(object_points, rvec_rod, trans_right, camera_matrix_, dist_coef_, r_projected_points);

    cv::Point2f l_proj_min = l_projected_points[0];
    cv::Point2f r_proj_min = r_projected_points[0];

    // Maximum disparity
    max_disparity_ = r_proj_min.x - l_proj_min.x;
  }

  void Feature::undistortKeyPoints(const vector<cv::KeyPoint> in, vector<cv::KeyPoint>& out)
  {
    if(dist_coef_.at<float>(0)==0.0)
    {
      out = in;
      return;
    }

    // Fill matrix with points
    cv::Mat u_kps(in.size(), 2, CV_32F);
    for(unsigned int i=0; i<in.size(); i++)
    {
      u_kps.at<float>(i,0) = in[i].pt.x;
      u_kps.at<float>(i,1) = in[i].pt.y;
    }

    // Undistort points
    u_kps = u_kps.reshape(2);
    cv::undistortPoints(u_kps, u_kps, camera_matrix_, dist_coef_, cv::Mat(), camera_matrix_);
    u_kps = u_kps.reshape(1);

    // Fill undistorted keypoint vector
    out.resize(in.size());
    for(unsigned int i=0; i<in.size(); i++)
    {
      cv::KeyPoint kp = in[i];
      kp.pt.x = u_kps.at<float>(i,0);
      kp.pt.y = u_kps.at<float>(i,1);
      out[i] = kp;
    }
  }

  void Feature::ratioMatching(const cv::Mat desc_1, const cv::Mat desc_2, const double ratio, vector<cv::DMatch> &matches)
  {
    matches.clear();
    if (desc_1.rows < 10 || desc_2.rows < 10) return;

    cv::Mat match_mask;
    const int knn = 2;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher;
    descriptor_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    vector<vector<cv::DMatch> > knn_matches;
    descriptor_matcher->knnMatch(desc_1, desc_2, knn_matches, knn, match_mask);
    for (uint m=0; m<knn_matches.size(); m++)
    {
      if (knn_matches[m].size() < 2) continue;
      if (knn_matches[m][0].distance <= knn_matches[m][1].distance * ratio)
        matches.push_back(knn_matches[m][0]);
    }
  }

  int Feature::stereoMatchingFilter(const vector<cv::KeyPoint> ukp_1, const vector<cv::KeyPoint> ukp_2, const vector<cv::DMatch> matches, vector<cv::DMatch>& matches_filtered)
  {
    // TODO: Change the epipolar threshold depending on the distortion coefficients
    const float epi_thresh = 1.5;

    for (size_t i=0; i<matches.size(); ++i)
    {
      if ( (abs(ukp_1[matches[i].queryIdx].pt.y - ukp_2[matches[i].trainIdx].pt.y) < epi_thresh) &&
           (abs(ukp_1[matches[i].queryIdx].pt.x - ukp_2[matches[i].trainIdx].pt.x) < max_disparity_) )
        matches_filtered.push_back(matches[i]);
    }

    return (int)matches_filtered.size();
  }

}
