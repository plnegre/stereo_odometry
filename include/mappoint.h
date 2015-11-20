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
   * \param Descriptor corresponding to this point
   * \param Keypoint response
   */
  MapPoint(const cv::Point3d p, const cv::Mat desc, const float kp_resp);

  /** \brief Return the point world position
   * @return the point world position
   */
  inline cv::Point3d getWorldPos() const {return world_pos_;}

private:

  cv::Point3d world_pos_; //!> World position

  cv::Mat desc_; //!> Best descriptor

  float kp_resp_; //!> Keypoint response, to decide the best descriptor when this point is seen from different frames

  int n_visible_; //!> Number of times this point have been seen

};

} // namespace

#endif // MAPPOINT_H
