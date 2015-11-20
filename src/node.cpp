#include <ros/ros.h>

#include <boost/thread.hpp>

#include "tracking.h"

/** \brief Read the node parameters
  */
void readParameters(odom::Tracking::Params &tracking_params)
{
  ros::NodeHandle nhp("~");
  nhp.param("camera_name", tracking_params.camera_name, string(""));
}

/** \brief Main entry point
  */
int main(int argc, char **argv)
{
  // Override SIGINT handler
  ros::init(argc, argv, "stereo_odometry");
  ros::start();

  // Objects
  odom::Tracking tracker;

  // Read parameters
  odom::Tracking::Params tracking_params;
  readParameters(tracking_params);

  // Set the parameters for every object
  tracker.setParams(tracking_params);

  // Launch threads
  boost::thread trackingThread(&odom::Tracking::run, &tracker);

  // ROS spin
  ros::Rate r(10);
  while (ros::ok())
  {
    r.sleep();
  }

  ros::shutdown();

  return 0;
}