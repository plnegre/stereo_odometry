/**
 * @file
 * @brief The Map class is responsive to store the map points (presentation).
 */

#ifndef MAP_H
#define MAP_H

#include <ros/ros.h>

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

  /** \brief Add point to the map
   * \param Map point
   */
  void addMapPoint(MapPoint* mp);

  /** \brief Add a set of points to the map
   * \param Map points
   */
  void addMapPoints(vector<MapPoint*> mps);

private:

  set<MapPoint*> map_points_;

};

} // namespace

#endif // MAP_H
