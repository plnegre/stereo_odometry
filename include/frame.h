/**
 * @file
 * @brief The frame class represents each stereo camera frame (presentation).
 */

#ifndef FRAME_H
#define FRAME_H

#include <ros/ros.h>
#include <tf/transform_datatypes.h>

#include "featools.h"
#include "feature.h"

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
   * \param Featools object
   * \param Left stereo image
   * \param Right stereo image
   * \param Enter the first feature id and returns the last
   */
  Frame(Featools* feat, const cv::Mat l_img, const cv::Mat r_img, const uint frame_uid, uint& feat_id);

  /** \brief Draw stereo matches
   * @return the composed image with the matchings
   */
  cv::Mat drawStereoMatches();

  /** \brief Get the matched features
   * @return the features computed from the stereo matches
   */
  inline vector<Feature*> getFeatures() const {return features_;};

protected:

  /** \brief Computes world point
   * @return the world point
   * \param left keypoint
   * \param right keypoint
   */
  cv::Point3d computeWorldPoint(cv::KeyPoint l_kp, cv::KeyPoint r_kp);

private:

  Featools* featools_; //!> Features object

  cv::Mat l_img_; //!> Left image
  cv::Mat r_img_; //!> Right image

  vector<cv::KeyPoint> l_ukp_; //!> Left kps
  vector<cv::KeyPoint> r_ukp_; //!> Right kps

  vector<cv::DMatch> matches_filtered_; //!> Matches filtered

  vector<Feature*> features_; //!> Frame features vector

};

} // namespace

#endif // FRAME_H