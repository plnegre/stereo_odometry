/**
 * @file
 * @brief The tracker class is responsive to track consecutive image frames (presentation).
 */

#ifndef TRACKER_H
#define TRACKER_H

#include <ros/ros.h>
#include <std_msgs/Int8.h>
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
#include "5point.h"

using namespace std;

namespace odom
{

class Tracker
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
  Tracker();

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


  /** \brief Publish the stereo matches for the current frame
   * \param current frame
   */
  void publishStereoMatches(Frame frame);

  void publishNumTracked(const int num_tracked);

  /** \brief Search matches between current and previous frame
   * \param vector of pair matching indices
   */
  void searchMatches(vector< pair<uint, uint> >& c_p_matches);

  void estimateRotation(const vector< pair<uint, uint> > c_p_matches);

  int recoverPose(const cv::Mat E, const cv::Mat pts1, const cv::Mat pts2, cv::Mat& R, cv::Mat& t, cv::Mat& mask);

  void decomposeEssentialMat(const cv::Mat E_in, cv::Mat& R1, cv::Mat& R2, cv::Mat& t );

  void updateTracks(const vector< pair<uint, uint> > c_p_matches);

  void buildDescMat(const vector<Feature*> feat, cv::Mat& l_desc, cv::Mat& r_desc);


private:

  Frame c_frame_; //!> Current frame
  Frame p_frame_; //!> Previous frame

  Params params_; //!> Stores parameters.

  Featools* featools_; //!> Features object
  Optimizer* optimizer_; //!> Optimizer object

  cv::Mat camera_matrix_; //!> Camera matrix [fx 0 cx; 0 fy cy; 0 0 1]
  cv::Mat dist_coef_; //!> Distortion coefficients [k1, k2, p1, p2]
  double baseline_; //!> Stereo baseline (in meters)

  uint frame_uid_; //!> Unique frame identifier
  uint feat_uid_; //!> Unique feature identifier

  vector< Feature* > tracked_feat_; //!> Tracked features

  ros::Publisher stereo_matching_pub_; //!> Publish an image with the stereo matchings
  ros::Publisher num_tracked_pub_; //!> Publish the number of tracked features

  // Topic sync
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image> SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;

};

} // namespace

#endif // TRACKER_H
