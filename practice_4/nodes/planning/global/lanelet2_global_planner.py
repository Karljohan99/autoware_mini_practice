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
        self.goal_point = None
        self.current_location = None

        self.speed_limit = rospy.get_param("~speed_limit")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        self.output_frame = rospy.get_param("~output_frame")
        self.distance_to_goal_limit = rospy.get_param("~distance_to_goal_limit")

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
            return None

        # find shortest path
        path = route.shortestPath()
        # this returns LaneletSequence to a point where lane change would be necessary to continue
        path_no_lane_change = path.getRemainingLane(start_lanelet)

        waypoints = self.lanelet_sequence_to_waypoints(path_no_lane_change)

        self.publish_waypoints(waypoints)
        
        
    def current_location_callback(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        if self.goal_point is None:
            return
        
        dist = math.sqrt((self.current_location.x - self.goal_point.x) ** 2 + (self.current_location.y - self.goal_point.y) ** 2)
        
        if dist < self.distance_to_goal_limit:
            self.publish_waypoints([])
            self.goal_point = None
            rospy.loginfo("Goal distance limit reached. Path is cleared.")


    def lanelet_sequence_to_waypoints(self, lanelet_path):
        waypoints = []
        is_last_lanelet = False

        for i, lanelet in enumerate(lanelet_path):
            if i == len(lanelet_path) - 1:
                is_last_lanelet = True

            if 'speed_ref' in lanelet.attributes:
                speed = float(lanelet.attributes['speed_ref']) / 3.6
            else:
                speed = float(self.speed_limit)
            
            if is_last_lanelet:
                last_lanelet_centerline = LineString([[point.x, point.y] for point in lanelet.centerline])

                proj_dist = last_lanelet_centerline.project(Point(self.goal_point.x, self.goal_point.y))

                last_waypoint = last_lanelet_centerline.interpolate(proj_dist)

                for j, point in enumerate(lanelet.centerline):
                    waypoint = Waypoint()

                    if j == 0 or LineString(last_lanelet_centerline.coords[:j+1]).length < proj_dist:
                        waypoint.pose.pose.position.x = point.x
                        waypoint.pose.pose.position.y = point.y
                        waypoint.pose.pose.position.z = point.z
                        waypoint.twist.twist.linear.x = speed
                        waypoints.append(waypoint)
                    else:
                        first_trimmed_point = Point(point.x, point.y)
                        previous_point = Point(lanelet.centerline[j-1].x, lanelet.centerline[j-1].y)

                        old_dist = previous_point.distance(first_trimmed_point)
                        new_dist = previous_point.distance(last_waypoint)

                        last_waypoint_z = lanelet.centerline[j-1].z + (lanelet.centerline[j-1].z-point.z)*new_dist/old_dist

                        waypoint.pose.pose.position.x = last_waypoint.x
                        waypoint.pose.pose.position.y = last_waypoint.y
                        waypoint.pose.pose.position.z = last_waypoint_z
                        waypoint.twist.twist.linear.x = speed
                        waypoints.append(waypoint) 


            else:
                for point in lanelet.centerline:
                    # create Waypoint and get the coordinats from lanelet.centerline points
                    waypoint = Waypoint()
                    waypoint.pose.pose.position.x = point.x
                    waypoint.pose.pose.position.y = point.y
                    waypoint.pose.pose.position.z = point.z
                    waypoint.twist.twist.linear.x = speed
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