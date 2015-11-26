/**
 * @file
 * @brief The Map class is responsive to store the map points (presentation).
 */

#ifndef MAP_H
#define MAP_H

#include <ros/ros.h>
#include <tf/transform_datatypes.h>

#include "mappoint.h"

using namespace std;

namespace odom
{

class Map
{

public:

  /** \brief Class constructor
   */
  Map();

  /** \brief Set parameters
   * \param camera matrix
   * \param distortion coefficients
   */
  void setParameters(const cv::Mat camera_matrix, const cv::Mat dist_coef, const int img_w, const int img_h);

  /** \brief Add point to the map
   * \param Map point
   */
  void addMapPoint(MapPoint* mp);

  /** \brief Add a set of points to the map
   * \param Map points
   */
  void addMapPoints(vector<MapPoint*> mps);

  /** \brief Get the map points in frustum
   * @return Map points
   * \param Current camera pose in world coordinates
   */
  vector<MapPoint*> getMapPointsInFrustum(const tf::Transform camera_pose);

  /** \brief Get the point properties of a specific set of map points
   * \param Set of map points
   * \param Output vector of left keypoints
   * \param Output vector of right keypoints
   * \param Output vector of left descriptors
   * \param Output vector of right descriptors
   */
  void getPointsProperties(const vector<MapPoint*> mps, vector<cv::Point3d>& wps, vector<cv::KeyPoint>& l_kps, vector<cv::KeyPoint>& r_kps, cv::Mat& l_desc, cv::Mat& r_desc);

protected:

  /** \brief Project a 3D point to camera pixels
   * @return Camera pixels
   * \param 3D point
   * \param Camera position
   */
  cv::Point2d project3dToPixel(const cv::Point3d point, const tf::Transform camera_pose);

private:

  set<MapPoint*> map_points_; //!> Map points

  cv::Mat camera_matrix_; //!> Camera matrix [fx 0 cx; 0 fy cy; 0 0 1]
  cv::Mat dist_coef_; //!> Distortion coefficients [k1, k2, p1, p2]
  int img_w_; //!> Image width
  int img_h_; //!> Image height

};

} // namespace

#endif // MAP_H
