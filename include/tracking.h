/**
 * @file
 * @brief The tracking class is responsive to track consecutive image frames (presentation).
 */

#ifndef TRACKING_H
#define TRACKING_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
#include <tf/transform_datatypes.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "features.h"
#include "frame.h"
#include "optimizer.h"
#include "map.h"

using namespace std;

namespace odom
{

class Tracking
{

public:

  struct Params
  {
    string camera_name;

    Params () {
      camera_name = "/stereo";
    }
  };

  /** \brief Class constructor
   */
  Tracking();

  /** \brief Set class params
   * \param the parameters struct
   */
  inline void setParams(const Params& params){params_ = params;}

  /** \brief Get class params
   */
  inline Params getParams() const {return params_;}

  /** \brief Starts tracking
   */
  void run();

protected:

  /** \brief Messages callback. This function is called when synchronized image
   * messages are received.
   * \param l_img left stereo image message of type sensor_msgs::Image
   * \param r_img right stereo image message of type sensor_msgs::Image
   */
  void msgsCallback(const sensor_msgs::ImageConstPtr& l_img_msg,
                    const sensor_msgs::ImageConstPtr& r_img_msg);

  /** \brief Search for matches between visible map points and current frame points
   * \param Set of visible map points
   * \param Current frame
   * \param Output vector containing the indices of the matched world points
   * \param Output vector containing the indices of the matched keypoints
   */
  void searchMatches(const vector<MapPoint*> mps,
                     Frame frame,
                     vector<int>& matched_map_wps,
                     vector<int>& matched_frame_kps);

  /** \brief Builds the matched vectors to estimate the pose
   * \param Set of map points
   * \param Current frame
   * \param Indices of the matched world points
   * \param Indices of the matched keypoints
   * \param Output vector containing the matched world points
   * \param Output vector containing the left matched keypoints
   * \param Output vector containing the right matched keypoints
   */
  void buildMatchedVectors(const vector<MapPoint*> mps,
                           Frame frame,
                           const vector<int> matched_map_wps,
                           const vector<int> matched_frame_kps,
                           vector<cv::Point3d>& map_wps,
                           vector<cv::KeyPoint>& frame_l_kps,
                           vector<cv::KeyPoint>& frame_r_kps);

  /** \brief Estimate the camera position
   * \param Set of map points
   * \param Current frame
   * \param Indices of the matched world points
   * \param Indices of the matched keypoints
   * \param Output estimated camera pose
   * \param Vector of inliers
   */
  void estimatePose(const vector<MapPoint*> mps,
                   Frame frame,
                   vector<int> matched_map_wps,
                   vector<int> matched_frame_kps,
                   tf::Transform& pose,
                   vector<int>& inliers);

  /** \brief Compute the projection error
   * \param Camera pose
   * \param World point
   * \param Camera point
   * \param Baseline (if camera point is in the right frame)
   */
  double computeProjectionError(const tf::Transform pose,
                                const cv::Point3d wp,
                                const cv::Point2d cp,
                                const double baseline=0.0);

  /** \brief Update the map points
   * \param Set of map points
   * \param Current frame
   * \param Indices of the matched map points
   * \param Indices of the matched keypoints
   * \param Vector of inliers
   */
  void updateMap(const vector<MapPoint*> mps,
                 Frame frame,
                 const vector<int> matched_map_wps,
                 const vector<int> matched_frame_kps,
                 const vector<int> inliers);

  /** \brief Publish the stereo matches for the current frame
   * \param current frame
   */
  void publishStereoMatches(Frame frame);


private:

  Params params_; //!> Stores parameters.

  Feature* feat_; //!> Features object

  Optimizer* optimizer_; //!> Optimizer object

  Map* map_; //!> Map object

  bool first_; //!> First iteration

  cv::Mat camera_matrix_; //!> Camera matrix [fx 0 cx; 0 fy cy; 0 0 1]
  cv::Mat dist_coef_; //!> Distortion coefficients [k1, k2, p1, p2]
  double baseline_; //!> Stereo baseline (in meters)
  int img_w_; //!> Image width
  int img_h_; //!> Image height

  ros::Publisher stereo_matching_pub_; //!> Publish an image with the stereo matchings

  ros::Publisher pose_pub_; //!> Publish vehicle pose

  tf::Transform camera_pose_; //!> Current camera pose

  // Topic sync
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image> SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;

};

} // namespace

#endif // TRACKING_H
