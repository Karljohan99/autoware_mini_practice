#!/usr/bin/env python3
import math
import rospy

from shapely.geometry import Point, LineString

from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import Waypoint
from autoware_msgs.msg import Lane

import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest


class Lanelet2GlobalPlanner:

    def __init__(self):

        # Parameters
        self.speed_limit = rospy.get_param("~speed_limit")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        self.output_frame = rospy.get_param("~output_frame")
        self.distance_to_goal_limit = rospy.get_param("~distance_to_goal_limit")

        # Global variables
        self.goal_point = None
        self.current_location = None
        self.last_waypoint = None

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        self.lanelet2_map = load(lanelet2_map_name, projector)

        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        #Publishers
        self.waypoints_pub = rospy.Publisher('/global_path', Lane, latch=True)


        #Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_point_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_location_callback, queue_size=1)


    def goal_point_callback(self, msg):
        if self.current_location is None:
            rospy.logwarn("Current location is not available!")
            return

        self.goal_point =  BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        # loginfo message about receiving the goal point
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                            msg.pose.orientation.w, msg.header.frame_id)

        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
        # find routing graph
        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)

        if route is None:
            rospy.logwarn("No route found!")
            return

        # find shortest path
        path = route.shortestPath()
        # this returns LaneletSequence to a point where lane change would be necessary to continue
        path_no_lane_change = path.getRemainingLane(start_lanelet)

        waypoints = self.lanelet_sequence_to_waypoints(path_no_lane_change)

        self.publish_waypoints(waypoints)


    def current_location_callback(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        if self.last_waypoint is None:
            return

        dist = math.sqrt((self.current_location.x - self.last_waypoint.x) ** 2 + (self.current_location.y - self.last_waypoint.y) ** 2)

        # check if goal reached
        if dist < self.distance_to_goal_limit:
            self.publish_waypoints([])
            self.goal_point = None
            self.last_waypoint = None
            rospy.loginfo("Goal distance limit reached. Path is cleared.")


    def lanelet_sequence_to_waypoints(self, lanelet_path):
        waypoints = []
        is_last_lanelet = False

        for i, lanelet in enumerate(lanelet_path):
            if i == len(lanelet_path) - 1:
                is_last_lanelet = True
                last_lanelet_centerline = LineString([[point.x, point.y] for point in lanelet.centerline])
                proj_dist = last_lanelet_centerline.project(Point(self.goal_point.x, self.goal_point.y))
                last_waypoint = last_lanelet_centerline.interpolate(proj_dist)

            speed = float(self.speed_limit)
            if 'speed_ref' in lanelet.attributes:
                speed = min(speed, float(lanelet.attributes['speed_ref']))
            speed = speed / 3.6


            for j, point in enumerate(lanelet.centerline):
                waypoint = Waypoint()
                waypoint.twist.twist.linear.x = speed

                if not is_last_lanelet or is_last_lanelet and (j == 0 or LineString(last_lanelet_centerline.coords[:j+1]).length < proj_dist):
                    if j >= len(lanelet.centerline) - 1:
                        continue

                    waypoint.pose.pose.position.x = point.x
                    waypoint.pose.pose.position.y = point.y
                    waypoint.pose.pose.position.z = point.z

                # if last lanelet then trim the centerline at nearest point before the goal point and add the goal point as the last point
                else:
                    first_trimmed_point = Point(point.x, point.y)
                    previous_point = Point(lanelet.centerline[j-1].x, lanelet.centerline[j-1].y)

                    old_dist = previous_point.distance(first_trimmed_point)
                    new_dist = previous_point.distance(last_waypoint)

                    waypoint.pose.pose.position.x = last_waypoint.x
                    waypoint.pose.pose.position.y = last_waypoint.y

                    # calculate z-coordinate for the goal point assuming linear change in elevation
                    waypoint.pose.pose.position.z  = lanelet.centerline[j-1].z + (lanelet.centerline[j-1].z-point.z)*new_dist/old_dist

                    self.last_waypoint = BasicPoint2d(last_waypoint.x, last_waypoint.y)

                waypoints.append(waypoint)

        return waypoints


    def publish_waypoints(self, waypoints):
        lane = Lane()
        lane.header.frame_id = self.output_frame
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        self.waypoints_pub.publish(lane)


    def run(self):
        rospy.spin()



if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()