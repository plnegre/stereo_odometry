/**
 * @file
 * @brief The Feature class is responsive to store the information of a map point (presentation).
 */

#ifndef FEATURE_H
#define FEATURE_H

#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <tf/transform_datatypes.h>

using namespace std;

namespace odom
{

class Feature
{

public:

  /** \brief Class constructor
   */
  Feature();

  /** \brief Class constructor
   * \param feature unique identifier
   * \param frame unique identifier
   * \param World position
   * \param Left frame keypoint
   * \param Right frame keypoint
   * \param Left frame descriptor
   * \param Right frame descriptor
   */
  Feature(const uint feat_uid, const uint frame_uid, const cv::Point3d world_pos, const cv::KeyPoint l_kp, const cv::KeyPoint r_kp, const cv::Mat l_desc, const cv::Mat r_desc);

  /** \brief Add another view of this feature
   * \param feature unique identifier
   * \param frame unique identifier
   * \param World position
   * \param Left frame keypoint
   * \param Right frame keypoint
   * \param Left frame descriptor
   * \param Right frame descriptor
   */
  void addView(const uint feat_uid, const uint frame_uid, const cv::Point3d world_pos, const cv::KeyPoint l_kp, const cv::KeyPoint r_kp, const cv::Mat l_desc, const cv::Mat r_desc);

  /** \brief Add another view of this feature without descriptors to reduce the memory usage
   * \param feature unique identifier
   * \param frame unique identifier
   * \param World position
   * \param Left frame keypoint
   * \param Right frame keypoint
   */
  void addView(const uint feat_uid, const uint frame_uid, const cv::Point3d world_pos, const cv::KeyPoint l_kp, const cv::KeyPoint r_kp);


  /** \brief Return the last point world position
   * @return the point world position
   */
  inline cv::Point3d getLastWorldPos() const {return world_pos_[world_pos_.size()-1];}

  /** \brief Return the last left keypoint
   * @return the left keypoint
   */
  inline cv::KeyPoint getLastLeftKp() const {return l_kp_[l_kp_.size()-1];}

  /** \brief Return the last right keypoint
   * @return the right keypoint
   */
  inline cv::KeyPoint getLastRightKp() const {return r_kp_[r_kp_.size()-1];}

  /** \brief Return the last left descriptor
   * @return the left descriptor
   */
  inline cv::Mat getLastLeftDesc() const {return l_desc_[l_desc_.size()-1];}

  /** \brief Return the last right descriptor
   * @return the right descriptor
   */
  inline cv::Mat getLastRightDesc() const {return r_desc_[r_desc_.size()-1];}

  /** \brief Return the last uid
   * @return the last uid
   */
  inline uint getLastUid() const {return uids_[uids_.size()-1];}

  /** \brief Return the newest frame uid
   * @return the newest frame uid
   */
  inline uint getNewestFrameUid() const {return newest_frame_uid_;}

  /** \brief Return vector of uids
   * @return the vector of uids
   */
  inline vector<uint> getUids() const {return uids_;}


private:

  vector<uint> uids_; //!> Unique identifiers

  uint newest_frame_uid_; //!> The newest frame uid

  vector<cv::Point3d> world_pos_; //!> World position relative to the camera frame

  vector<cv::KeyPoint> l_kp_; //!> Left keypoint

  vector<cv::KeyPoint> r_kp_; //!> Right keypoint

  vector<cv::Mat> l_desc_; //!> Left descriptor

  vector<cv::Mat> r_desc_; //!> Right descriptor

};

} // namespace

#endif // FEATURE_H
