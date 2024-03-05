#!/usr/bin/env python3

import rospy

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

        waypoints = lanelet_sequence_to_waypoints(path_no_lane_change, self.speed_limit)

        publish_wayspoints(waypoints, self.waypoints_pub, self.output_frame)
        
        
    def current_location_callback(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)


    def run(self):
        rospy.spin()


def lanelet_sequence_to_waypoints(lanelet_path, speed_limit):
    waypoints = []
    for lanelet in lanelet_path:
        if 'speed_ref' in lanelet.attributes:
            speed = float(lanelet.attributes['speed_ref']) / 3.6
        else:
            speed = float(speed_limit)
        
        for point in lanelet.centerline:
            # create Waypoint and get the coordinats from lanelet.centerline points
            waypoint = Waypoint()
            waypoint.pose.pose.position.x = point.x
            waypoint.pose.pose.position.y = point.y
            waypoint.pose.pose.position.z = point.z
            waypoint.twist.twist.linear.x = speed
            waypoints.append(waypoint)
        
    return waypoints


def publish_wayspoints(waypoints, waypoints_pub, output_frame):
    lane = Lane()        
    lane.header.frame_id = output_frame
    lane.header.stamp = rospy.Time.now()
    lane.waypoints = waypoints
    waypoints_pub.publish(lane)





if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()
