/**
 * @file
 * @brief The features class represents the features of a frame (presentation).
 */

#ifndef FEATOOLS_H
#define FEATOOLS_H

#include <ros/ros.h>
#include <tf/transform_datatypes.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;

namespace odom
{

class Featools
{

public:

  /** \brief Empty class constructor
   */
  Featools();

  /** \brief Set parameters
   * \param Camera matrix
   * \param Distortion coefficients
   * \param baseline
   */
  void setParameters(const cv::Mat camera_matrix, const cv::Mat dist_coef, const double baseline);

  /** \brief Undistort keypoints
   * \param Input distorted keypoints
   * \param Output undistorted keypoints
   * \param Camera matrix
   * \param Distortion coefficients
   */
  void undistortKeyPoints(const vector<cv::KeyPoint> in, vector<cv::KeyPoint>& out);

  /** \brief Performs ration matching between two set of descriptors
   * \param Input descriptors 1
   * \param Input descriptors 2
   * \param Ratio (0.6-0.8)
   * \param Output matches
   */
  void ratioMatching(const cv::Mat desc_1,
                     const cv::Mat desc_2,
                     const double ratio,
                     vector<cv::DMatch> &matches);

  /** \brief Filters the matches between two stereo images
   * \param Input undistorted keypoints 1
   * \param Input undistorted keypoints 2
   * \param Vector of descriptor matchings
   * \param Output vector of filtered descriptor matchings
   */
  int stereoMatchingFilter(const vector<cv::KeyPoint> ukp_1,
                           const vector<cv::KeyPoint> ukp_2,
                           const vector<cv::DMatch> matches,
                           vector<cv::DMatch>& matches_filtered);

  /** \brief Get the camera matrix
   * @return the camera matrix
   */
  inline cv::Mat getCameraMatrix() const {return camera_matrix_;};

  /** \brief Get the distortion coefficients
   * @return the distortion coefficients
   */
  inline cv::Mat getDistCoef() const {return dist_coef_;};

  /** \brief Get the baseline
   * @return the baseline
   */
  inline double getBaseline() const {return baseline_;};

private:

  cv::Mat camera_matrix_; //!> Camera matrix [fx 0 cx; 0 fy cy; 0 0 1]
  cv::Mat dist_coef_; //!> Distortion coefficients [k1, k2, p1, p2]
  double baseline_; //!> Stereo baseline (in meters)

  double max_disparity_; //!> Maximum disparity

};

} // namespace

#endif // FEATOOLS_H