/**
 * @file
 * @brief The frame class represents each stereo camera frame (presentation).
 */

#ifndef FRAME_H
#define FRAME_H

#include <ros/ros.h>
#include <tf/transform_datatypes.h>

#include "feature.h"
#include "mappoint.h"

using namespace std;

namespace odom
{

class Frame
{

public:

  /** \brief Empty class constructor
   */
  Frame();

  /** \brief Class constructor
   * \param Left stereo image
   * \param Right stereo image
   * \param Camera matrix
   * \param Distortion coefficients
   * \param baseline
   */
  Frame(const cv::Mat l_img, const cv::Mat r_img, const cv::Mat camera_matrix, const cv::Mat dist_coef, const double baseline);

  /** \brief Get the left stereo matched keypoints
   * @return the vector of left matched keypoints
   */
  vector<cv::KeyPoint> getLeftMatchedKp();

  /** \brief Get the right stereo matched keypoints
   * @return the vector of right matched keypoints
   */
  vector<cv::KeyPoint> getRightMatchedKp();

  /** \brief Get the left stereo matched descriptors
   * @return the matched descriptors matrix
   */
  cv::Mat getLeftMatchedDesc();

  /** \brief Get the right stereo matched descriptors
   * @return the matched descriptors matrix
   */
  cv::Mat getRightMatchedDesc();

  /** \brief Get the matched map points
   * @return the map points computed from the stereo matches
   */
  vector<MapPoint*> getMapPoints();

  /** \brief Draw stereo matches
   * @return the composed image with the matchings
   */
  cv::Mat drawMatches();


private:

  Feature* feat_; //!> Features object

  cv::Mat l_img_; //!> Left image
  cv::Mat r_img_; //!> Right image

  vector<cv::KeyPoint> l_kp_; //!> Left keypoints.
  vector<cv::KeyPoint> r_kp_; //!> Right keypoints.

  cv::Mat l_desc_; //!> Left descriptors.
  cv::Mat r_desc_; //!> Right descriptors.

  vector<cv::DMatch> matches_filtered_; //!> Stereo filtered matches

  cv::Mat camera_matrix_; //!> Camera matrix [fx 0 cx; 0 fy cy; 0 0 1]
  cv::Mat dist_coef_; //!> Distortion coefficients [k1, k2, p1, p2]
  double baseline_; //!> Stereo baseline (in meters)

};

} // namespace

#endif // FRAME_H