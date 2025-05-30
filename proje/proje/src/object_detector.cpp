#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_listener.h>
#include <ros/package.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <zbar.h>
#include <map>
#include <set>
#include <dirent.h>
#include <tf/transform_listener.h>

ros::Publisher marker_pub;
image_transport::Publisher debug_image_pub;
cv::Mat latest_depth;
bool depth_received = false;

nav_msgs::Odometry latest_odom;
bool odom_received = false;
tf::TransformListener* tf_listener = nullptr;

cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
std::map<std::string, std::vector<cv::KeyPoint>> ref_keypoints;
std::map<std::string, cv::Mat> ref_descriptors;
double fx = 476.703, cx = 400.5,cy = 240.5;
int image_width = 640;
double camera_fov_rad = 2 * atan(image_width / (2.0 * fx));
double camera_fov_deg = camera_fov_rad * 180.0 / M_PI;
std::set<std::pair<int, int>> seen_positions;
const double POSITION_TOLERANCE = 0.3;

double getYawFromQuaternion(const geometry_msgs::Quaternion& q) {
    tf::Quaternion tf_q(q.x, q.y, q.z, q.w);
    double roll, pitch, yaw;
    tf::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    return yaw;
}

double pixelToAngle(int u) {
    double norm = (u - cx) / double(cx);
    return norm * (camera_fov_deg / 2.0) * M_PI / 180.0;
}

bool isDuplicate(double x, double y) {
    for (const auto& pos : seen_positions) {
        double dx = (pos.first * POSITION_TOLERANCE) - x;
        double dy = (pos.second * POSITION_TOLERANCE) - y;
        if (std::sqrt(dx*dx + dy*dy) < POSITION_TOLERANCE)
            return true;
    }
    int xi = static_cast<int>(x / POSITION_TOLERANCE);
    int yi = static_cast<int>(y / POSITION_TOLERANCE);
    seen_positions.insert(std::make_pair(xi, yi));
    return false;
}

void publishMarker(const std::string& label, float r, float g, float b, double x, double y,const ros::Time& stamp, bool text = true) {
    static int id = 0;
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time(0);
    marker.ns = label;
    marker.id = id++;
    marker.type = text ? visualization_msgs::Marker::TEXT_VIEW_FACING : visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = text ? 1.0 : 0.2;
    marker.scale.x = text ? 0.0 : 0.3;
    marker.scale.y = text ? 0.0 : 0.3;
    marker.scale.z = text ? 0.5 : 0.5;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 1.0;
    marker.text = text ? label : "";
    marker_pub.publish(marker);
}

std::string classifyHazmat(const cv::Mat& roi) {
    if (roi.empty()) return "Unknown";

    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    akaze->detectAndCompute(gray, cv::noArray(), kps, desc);

    std::string best_label = "Unknown";
    int best_matches = 0;

    for (const auto& [label, ref_desc] : ref_descriptors) {
        if (ref_desc.empty() || desc.empty()) continue;

        std::vector<std::vector<cv::DMatch>> knn_matches;
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.knnMatch(desc, ref_desc, knn_matches, 2);

        int good_matches = 0;
        for (const auto& m : knn_matches) {
            if (m.size() == 2 && m[0].distance < 0.5 * m[1].distance) {
                good_matches++;
            }
        }

        if (good_matches > best_matches) {
            best_matches = good_matches;
            best_label = label;
        }
    }

    if (best_matches < 3) return "Unknown";
    return best_label;
}



bool transformCameraPointToMap(double X, double Y, double Z, const ros::Time& stamp, double& out_x, double& out_y) {
    geometry_msgs::PointStamped pt_cam, pt_map;
    pt_cam.header.frame_id = "p3at/camera_depth_optical_frame";
    pt_cam.header.stamp = ros::Time(0);
    pt_cam.point.x = X;
    pt_cam.point.y = Y;
    pt_cam.point.z = Z;


    if (!tf_listener->waitForTransform("map", pt_cam.header.frame_id, stamp, ros::Duration(0.3))) {
        ROS_WARN("TF hazır değil (zaman uyuşmazlığı)!");
        return false;
    }


    try {
        tf_listener->transformPoint("map", pt_cam, pt_map);
        out_x = pt_map.point.x;
        out_y = pt_map.point.y;
        return true;
    } catch (tf::TransformException& ex) {
        ROS_WARN("TF dönüşüm hatası: %s", ex.what());
        return false;
    }
}


void detectObjects(const cv::Mat& bgr, const ros::Time& stamp, cv::Mat& debug_image) {
	ROS_INFO_THROTTLE(2.0, "detectObjects çalışıyor");
   if (!odom_received || !depth_received) return;

	double linear_speed = std::abs(latest_odom.twist.twist.linear.x);
   double angular_speed = std::abs(latest_odom.twist.twist.angular.z);
   
   const double SPEED_THRESHOLD = 0.05;  // 5 cm/s'ten küçükse duruyor say
   
   if (linear_speed > SPEED_THRESHOLD || angular_speed > SPEED_THRESHOLD) {
       // Robot hareket ediyor, tespit yapma
       ROS_INFO_THROTTLE(1.0, "Robot hareket ediyor, obje tespiti yapılmıyor.");
       return;
   }
    // --- HAZMAT ---

    cv::Mat gray, thresh;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, thresh, 100, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 500.0) continue;

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);
        if (approx.size() != 4) continue;

        cv::Rect box = cv::boundingRect(approx);
        if (box.x < 0 || box.y < 0 || box.x + box.width > bgr.cols || box.y + box.height > bgr.rows) continue;

        cv::RotatedRect minRect = cv::minAreaRect(approx);
	cv::Point2f srcPoints[4];
	minRect.points(srcPoints);

	float width = minRect.size.width;
	float height = minRect.size.height;
	if (width < height) std::swap(width, height);

	cv::Point2f dstPoints[4] = {
	    cv::Point2f(0, height - 1),
	    cv::Point2f(0, 0),
	    cv::Point2f(width - 1, 0),
	    cv::Point2f(width - 1, height - 1)
	};

	cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
	cv::Mat flatHazmat;
	cv::warpPerspective(bgr, flatHazmat, perspectiveMatrix, cv::Size(width, height));
	cv::Mat rot = flatHazmat;


        std::string label = classifyHazmat(rot);
        if (label == "Unknown") continue;
        cv::Point2f vertices[4];
	minRect.points(vertices);
        for (int i = 0; i < 4; ++i)
	    cv::line(debug_image, vertices[i], vertices[(i+1)%4], cv::Scalar(255, 0, 0), 2);
	cv::putText(debug_image, label, vertices[1], cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 0, 0), 2);
	
	cv::Rect safe_box = box;
	safe_box.x += box.width * 0.05;
	safe_box.y += box.height * 0.05;
	safe_box.width *= 0.8;
	safe_box.height *= 0.8;

        int v = safe_box.y + safe_box.height / 2;  // Nesnenin ortasına göre satır al
	int u = safe_box.x + safe_box.width / 2;   // Bu zaten vardı
	/*cv::Moments M = cv::moments(contour);
	int u = M.m10 / M.m00;
	int v = M.m01 / M.m00;*/

	if (!depth_received || u < 0 || u >= latest_depth.cols || v < 0 || v >= latest_depth.rows) continue;

	float Z = latest_depth.at<float>(v, u);
	if (!std::isfinite(Z) || Z < 0.1 || Z > 10.0) continue;  // min-max filtreleme ekle


	float X = (u - cx) * Z / fx;
	float Y = (v - cy) * Z / fx;

	double x, y;
	if (!transformCameraPointToMap(X, Y, Z, stamp, x, y)) continue;


        //if (isDuplicate(x, y)) continue;

        /*cv::Point2f vertices[4];
	minRect.points(vertices);
	for (int i = 0; i < 4; ++i)
	    cv::line(debug_image, vertices[i], vertices[(i+1)%4], cv::Scalar(255, 0, 0), 2);
	cv::putText(debug_image, label, vertices[1], cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 0, 0), 2);*/
	//ROS_INFO("Hazmat:%s",label);

        publishMarker(label, 1.0, 1.0, 1.0, x, y,stamp);
    }
    // --- QR ---
    zbar::ImageScanner scanner;
    scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);
    cv::Mat qrgray;
    cv::cvtColor(bgr, qrgray, cv::COLOR_BGR2GRAY);
    zbar::Image image(qrgray.cols, qrgray.rows, "Y800", qrgray.data, qrgray.cols * qrgray.rows);
    scanner.scan(image);
    for (auto symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
        std::string data = symbol->get_data();
        std::vector<cv::Point> pts;
        for (int i = 0; i < symbol->get_location_size(); i++)
            pts.emplace_back(symbol->get_location_x(i), symbol->get_location_y(i));
        cv::Rect box = cv::boundingRect(pts);
        cv::rectangle(debug_image, box, cv::Scalar(0,255,0), 2);
        cv::putText(debug_image, data, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
        int v = box.y + box.height / 2;  // Nesnenin ortasına göre satır al
	int u = box.x + box.width / 2;   // Bu zaten vardı
	if (!depth_received || u < 0 || u >= latest_depth.cols || v < 0 || v >= latest_depth.rows) continue;

	float Z = latest_depth.at<float>(v, u);
	if (!std::isfinite(Z) || Z < 0.2 || Z > 10.0) continue;  // min-max filtreleme ekle


	float X = (u - cx) * Z / fx;
	float Y = (v - cy) * Z / fx;

	double x, y;
	if (!transformCameraPointToMap(X, Y, Z, stamp, x, y)) continue;


        if (isDuplicate(x, y)) continue;
        //cv::rectangle(debug_image, box, cv::Scalar(0,255,0), 2);
        //cv::putText(debug_image, data, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
        //ROS_INFO("QR:%s",data);
        publishMarker(data, 0.0, 1.0, 0.0, x, y,stamp);
    }

    // --- VARIL --- (renk maskeleme)
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    std::vector<std::tuple<std::string, cv::Scalar, cv::Scalar, cv::Scalar>> renkler = {
        {"Yavruağzı", cv::Scalar(0,0,148), cv::Scalar(8,124,220), cv::Scalar(255,150,128)},
        {"Taba", cv::Scalar(5,172,91), cv::Scalar(179,255,142), cv::Scalar(153,76,0)},
        {"Siklamen", cv::Scalar(28,25,133), cv::Scalar(179,255,170), cv::Scalar(238,130,238)}
    };

    for (const auto& [isim, low, high, renk] : renkler) {
        cv::Mat mask, res;
        cv::inRange(hsv, low, high, mask);
        std::vector<std::vector<cv::Point>> conts;
        cv::findContours(mask.clone(), conts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& c : conts) {
            if (cv::contourArea(c) < 500.0) continue;
            cv::Moments m = cv::moments(c);
            int u = static_cast<int>(m.m10 / m.m00);
            int v = static_cast<int>(m.m01 / m.m00);
 
	     cv::Rect box = cv::boundingRect(c);
            cv::rectangle(debug_image, box, renk, 2);
            cv::putText(debug_image, isim, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, renk, 2);
	
	if (!depth_received || u < 0 || u >= latest_depth.cols || v < 0 || v >= latest_depth.rows) continue;

	cv::Rect region(u - 1, v - 1, 3, 3);
	region &= cv::Rect(0, 0, latest_depth.cols, latest_depth.rows);

	float Z = 0;
	int count = 0;
	for (int i = region.y; i < region.y + region.height; ++i) {
	    for (int j = region.x; j < region.x + region.width; ++j) {
		float d = latest_depth.at<float>(i, j);
		if (std::isfinite(d) && d > 0.2 && d < 10.0) {
		    Z += d;
		    count++;
		}
	    }
	}
	if (count == 0) continue;
	Z /= count;

	//if (!std::isfinite(Z) || Z < 0.2 || Z > 10.0) continue;  // min-max filtreleme ekle


	float X = (u - cx) * Z / fx;
	float Y = (v - cy) * Z / fx;

	double x, y;
	if (!transformCameraPointToMap(X, Y, Z, stamp, x, y)) continue;


            if (isDuplicate(X, Y)) continue;
            /*cv::Rect box = cv::boundingRect(c);
            cv::rectangle(debug_image, box, renk, 2);
            cv::putText(debug_image, isim, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, renk, 2);*/
           // ROS_INFO("Varil:%s",isim);
            publishMarker(isim, renk[2]/255.0, renk[1]/255.0, renk[0]/255.0, x, y,stamp);
        }
    }
}



void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv::Mat bgr;
    try { bgr = cv_bridge::toCvShare(msg, "bgr8")->image; } catch (...) { return; }
    cv::Mat debug_image = bgr.clone();
    detectObjects(bgr, msg->header.stamp, debug_image);
    auto dbg_msg = cv_bridge::CvImage(msg->header, "bgr8", debug_image).toImageMsg();
    debug_image_pub.publish(dbg_msg);
}
void depthCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        latest_depth = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
        depth_received = true;
        ROS_INFO_THROTTLE(1.0, "Derinlik alındı.");
    } catch (...) {
        ROS_WARN("Depth image conversion failed.");
        depth_received = false;
    }
}


void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) { latest_odom = *msg; odom_received = true; }

void loadHazmatReferences(const std::string& dir_path) {
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) {
        ROS_ERROR("Klasör açma hatası: %s", dir_path.c_str());
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string fname = entry->d_name;
        if (fname == "." || fname == "..") continue;
        std::string full_path = dir_path + "/" + fname;
        std::string label = fname.substr(0, fname.find_last_of('.'));

        cv::Mat img = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        cv::Point2f center(img.cols/2.0, img.rows/2.0);
	cv::Mat rot_mat = cv::getRotationMatrix2D(center, 45.0, 1.0);
	cv::warpAffine(img, img, rot_mat, img.size(), cv::INTER_LINEAR);
        akaze->detectAndCompute(img, cv::noArray(), kps, desc);
        if (!desc.empty()) {
            ref_keypoints[label] = kps;
            ref_descriptors[label] = desc;
            ROS_INFO("Yüklendi (AKAZE): %s", label.c_str());
        }
    }
    closedir(dir);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "hazmat_varil_qr_explorer");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    tf_listener = new tf::TransformListener();

    std::string pkg_path = ros::package::getPath("proje") + "/src/hazmats";
    loadHazmatReferences(pkg_path);
	
    marker_pub = nh.advertise<visualization_msgs::Marker>("detected_objects", 10);
    debug_image_pub = it.advertise("/debug/image", 1);

    image_transport::Subscriber image_sub = it.subscribe("/p3at/camera/rgb/image_raw", 1, imageCallback);
    image_transport::Subscriber depth_sub = it.subscribe("/p3at/p3at/camera/depth/image_raw", 1, depthCallback);
    ros::Subscriber odom_sub = nh.subscribe("/p3at/odom", 10, odomCallback);

    ros::spin();
    return 0;
}