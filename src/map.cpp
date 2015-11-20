#include "map.h"

namespace odom
{
  Map::Map(){}

  void Map::addMapPoint(MapPoint *mp)
  {
    map_points_.insert(mp);
  }

  void Map::addMapPoints(vector<MapPoint*> mps)
  {
    for (uint i=0; i<mps.size(); i++)
      addMapPoint(mps[i]);
  }

} //namespace odom