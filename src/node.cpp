#include <ros/ros.h>

#include <boost/thread.hpp>

#include "tracker.h"

/** \brief Read the node parameters
  */
void readParameters(odom::Tracker::Params &tracker_params)
{
  ros::NodeHandle nhp("~");
  nhp.param("camera_name", tracker_params.camera_name, string(""));
}

/** \brief Main entry point
  */
int main(int argc, char **argv)
{
  // Override SIGINT handler
  ros::init(argc, argv, "stereo_odometry");
  ros::start();

  // Objects
  odom::Tracker tracker;

  // Read parameters
  odom::Tracker::Params tracker_params;
  readParameters(tracker_params);

  // Set the parameters for every object
  tracker.setParams(tracker_params);

  // Launch threads
  boost::thread trackingThread(&odom::Tracker::run, &tracker);

  // ROS spin
  ros::Rate r(10);
  while (ros::ok())
  {
    r.sleep();
  }

  ros::shutdown();

  return 0;
}