#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include <visualization_msgs/Marker.h>

visualization_msgs::Marker trajectory;
ros::Publisher traj_pub;

void odomCb(const nav_msgs::Odometry::ConstPtr& odom_msg){
	trajectory.header.stamp = ros::Time::now();
	trajectory.points.push_back(odom_msg->pose.pose.position);
	traj_pub.publish(trajectory);
}

int main(int argc, char **argv){
	ros::init(argc, argv, "Gezinge");
	ros::NodeHandle nh;
	
	trajectory.header.frame_id = "p3at/trajectory";
	trajectory.ns = "traj";
	trajectory.action = visualization_msgs::Marker::ADD;
	trajectory.pose.orientation.w = 1.0;
	trajectory.id = 0;

	trajectory.type = visualization_msgs::Marker::LINE_STRIP;
	trajectory.scale.x = 0.1;
	trajectory.color.r = 0.0;
	trajectory.color.g = 1.0;
	trajectory.color.b = 1.0;
	trajectory.color.a = 1.0;
	
	
	ros::Subscriber odom_sub = nh.subscribe("/p3at/odom", 1000, odomCb);
	
	traj_pub = nh.advertise<visualization_msgs::Marker>("/p3at/traj", 1000);
	
	ros::spin();

	return 0;
}
