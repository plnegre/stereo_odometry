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

  /** \brief Search for matches between current and previous frame
   * \param Matched left keypoints on the previous frame
   * \param Matched right keypoints on the previous frame
   * \param Matched world points on the current frame
   */
  void searchMatches(vector<cv::Point2d>& matched_prev_kp_l, vector<cv::Point2d>& matched_prev_kp_r, vector<cv::Point3d>& matched_c_wp);

  /** \brief Estimate the motion between two frames
   * @return Number of inliers
   * \param Matched left keypoints on the previous frame
   * \param Matched right keypoints on the previous frame
   * \param Matched world points on the current frame
   * \param Output motion
   */
  int estimateMotion(vector<cv::Point2d> matched_prev_kp_l, vector<cv::Point2d> matched_prev_kp_r, vector<cv::Point3d> matched_wp, tf::Transform& delta);

  /** \brief Compute projection error
   * @return error
   * \param estimated transform
   * \param world point
   * \param corresponding camera point
   * \param baseline (0 for left camera)
   */
  double computeProjectionError(const tf::Transform delta, const cv::Point3d wp, const cv::Point2d cp, double baseline = 0.0);

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

  ros::Publisher stereo_matching_pub_; //!> Publish an image with the stereo matchings

  ros::Publisher pose_pub_; //!> Publish vehicle pose

  Frame p_frame_; //!> Previous frame
  Frame c_frame_; //!> Current frame

  tf::Transform acc_pose_; //!> Accumulated pose

  // Topic sync
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image> SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;

};

} // namespace

#endif // TRACKING_H
