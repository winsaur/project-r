#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <cvaux.h>
class MessageFetcher
{
	private:
		//transport handles
		ros::NodeHandle nh_;
		image_transport::ImageTransport it_;
		image_transport::Subscriber depth_sub_;
		image_transport::Subscriber rgb_sub_;

		//pointers to raw Mats
		cv_bridge::CvImagePtr raw_rgb_ptr_;
		cv_bridge::CvImagePtr raw_depth_ptr_;

		void depthCb(const sensor_msgs::ImageConstPtr& msg);
		void rgbCb(const sensor_msgs::ImageConstPtr& msg);

		bool raw_updated_,depth_updated_;
		void convertMsgToCvImagePtr(const sensor_msgs::ImageConstPtr& msg, cv_bridge::CvImagePtr& raw_ptr);
	public:
		bool GetFrame(cv::Mat& rgb_frame, cv::Mat& depth_frame);
		MessageFetcher();
		~MessageFetcher();
};