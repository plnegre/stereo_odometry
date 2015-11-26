/**
 * @file
 * @brief The MapPoint class is responsive to store the information of a map point (presentation).
 */

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;

namespace odom
{

class MapPoint
{

public:

  /** \brief Class constructor
   */
  MapPoint();

  /** \brief Class constructor
   * \param World position
   * \param Left frame keypoint
   * \param Right frame keypoint
   * \param Left frame descriptor
   * \param Right frame descriptor
   */
  MapPoint(const cv::Point3d p, const cv::KeyPoint l_kp, const cv::KeyPoint r_kp, const cv::Mat l_desc, const cv::Mat r_desc);

  /** \brief Return the point world position
   * @return the point world position
   */
  inline cv::Point3d getWorldPos() const {return world_pos_;}

  /** \brief Return the left keypoint
   * @return the left keypoint
   */
  inline cv::KeyPoint getLeftKp() const {return l_kp_;}

  /** \brief Return the right keypoint
   * @return the right keypoint
   */
  inline cv::KeyPoint getRightKp() const {return r_kp_;}

  /** \brief Return the left descriptor
   * @return the left descriptor
   */
  inline cv::Mat getLeftDesc() const {return l_desc_;}

  /** \brief Return the right descriptor
   * @return the right descriptor
   */
  inline cv::Mat getRightDesc() const {return r_desc_;}

private:

  cv::Point3d world_pos_; //!> World position

  cv::KeyPoint l_kp_; //!> Left best keypoint

  cv::KeyPoint r_kp_; //!> Right best keypoint

  cv::Mat l_desc_; //!> Left best descriptor

  cv::Mat r_desc_; //!> Right best descriptor

  int n_visible_; //!> Number of times this point have been seen

};

} // namespace

#endif // MAPPOINT_H
