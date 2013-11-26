/*
 * BagOnPersonProcessor.cpp
    Copyright (C) 2013  Timothy Sweet

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

//std includes
#include <sstream>

//library includes
#include <ros/ros.h>

//project includes
#include "../include/Utility.h"
#include "../include/ProcessorNode.h"
#include "../include/EventCodes.h"
#include "../include/BlobDescriptor.h"

class BagOnPersonProcessor : public ProcessorNode {
public:
	BagOnPersonProcessor(const std::string& blob_topic,const std::string& event_topic) :
		input_stream_(blob_topic) {
		event_name_ = BagOnPersonName;
		event_code_ = BagOnPersonCode;
		output_topic_ = event_topic;
		init();
		std::cout<<"subscribed to: "<<input_stream_.getTopic()<<std::endl;
	}
	void callback(const ros::TimerEvent& event);

	void init();
private:
	DISALLOW_COPY_AND_ASSIGN(BagOnPersonProcessor);
	DISALLOW_DEFAULT_CONSTRUCTION(BagOnPersonProcessor);
	BlobDescriptorFetcher input_stream_;
	//TODO: combine these into a single data structure
	//a relationship between blob IDs and their past similarities
	std::map<int,std::vector<double> > similarities_;
	//a relationship between blob IDs and their first frame contour
	std::map<int,cv::Mat> templates_;
	//a relationship between blob IDs and their centroids
	std::map<int,cv::Point2f> centroids_;
	//a relation between blob IDs and their first-seen bounding boxes
	std::map<int,cv::Rect> initial_bounds_;
	//a relation between blob IDs and points on the template_ used for warpAffine
	Point2fVecMap src_points_;
};

int main(int argc, char* argv[]) {
	//connect to ros
	ros::init(argc, argv, "segmentation");
	//set logger level
	utility::setLoggerDebug();
	ros::NodeHandle handle("~");

	BagOnPersonProcessor processor("blobs_in","events_out");
	//loop at 60 hz just because that seems like a pretty good rate given
	//that the camera is running at 30 hz
	ros::Timer timer = handle.createTimer(ros::Duration(1./30.), &BagOnPersonProcessor::callback, &processor);

	ros::spin();

	return 0;
}

void BagOnPersonProcessor::init() {
	setupOutput();
}

void BagOnPersonProcessor::callback(const ros::TimerEvent& event) {
	//for each blob which was updated this frame
	if(input_stream_.blobs_updated_.size()==0) {
		return;
	}
	//TODO: get the size another way
	cv::Mat output(cv::Size(640,480),CV_8UC1,cv::Scalar(0));
	std::vector<int>::iterator blob_iterator = input_stream_.blobs_updated_.begin();
	for(;blob_iterator!=input_stream_.blobs_updated_.end();) {
		//just store the index as an int to make things easier
		int cursor = *blob_iterator;
		//erase it from the list and reset the iterator. This prevents
		//jumping past the end of the vector when incrementing the iterator
		input_stream_.blobs_updated_.erase(blob_iterator);
		//TODO: fix the race condition here in which an empty list which
		//has a blob addd to it after here will break the loop
		blob_iterator = input_stream_.blobs_updated_.begin();

		//update blob number cursor, look it up in the map

		//draw the contour on a mat
		//TODO:get the size another way
		cv::Mat blob_visual(cv::Size(640,480),CV_8UC1,cv::Scalar(0));
		ContourList temp_list;
		temp_list.push_back(input_stream_.data_[cursor]->contour_);
		cv::drawContours(blob_visual,temp_list,0,cv::Scalar(255),-1);


		//compute the body axis
		//option 1: try computing it as the center of mass
		 /// Get the moment
		  cv::Moments mu = cv::moments( input_stream_.data_[cursor]->contour_, false );

		  ///  Get the mass center
		  cv::Point2f mc( mu.m10/mu.m00 , mu.m01/mu.m00 );
		//option 2: try computing it as the horizontal projection(as in sum of vertical stacks of pixels)
		  std::vector<int> upward_projection;
		utility::getUpwardProjection(blob_visual,&upward_projection);
		int max=0;
		int max_at=0;
		for(int i=0;i<upward_projection.size();i++) {
			if(upward_projection[i] > max) {
				max = upward_projection[i];
				max_at=i;
			}
		}
		std::cout<<std::endl;
		//option 3: calculate the median line
		int median_at = utility::getHorizontalMedian(blob_visual);
		//draw the mass center axis
		//cv::line(blob_visual,cv::Point(mc.x,0),cv::Point(mc.x,480),cv::Scalar(128),2);
		//draw max upward projection axis
		cv::line(blob_visual,cv::Point(max_at,0),cv::Point(max_at,480),cv::Scalar(128+64),2);
		//draw median line
		//cv::line(blob_visual,cv::Point(median_at,0),cv::Point(median_at,480),cv::Scalar(128),2);

		int symmetry_axis = max_at;

		//classify each pixel as symmetric or asymmetric
		utility::recolorNonSymmetricRegions(symmetry_axis,&blob_visual);

		//check periodicity and reclassify pixels if necessary
		//if this is the first time the blob is being seen, set up its template
		//store the blob_visual for later comparison
		//store maximum similarity in the similarity tracker
		//TODO: make inherited class of BlobDescriptor which includes this stuff

		//get an image of just the blob
//		cv::Mat cropped_blob;
		//should I use CurrentBound (aka the filtered version) instead?
//		cv::Mat(blob_visual,input_stream_.data_[cursor]->LastRawBound()).copyTo(cropped_blob);
		if(input_stream_.data_[cursor]->FirstSeen()==input_stream_.data_[cursor]->LastSeen()) {

			//this is the first time the blob is seen: set up its template
			//this is guaranteed to make a new element because this is a new blob
			templates_.insert(std::pair<int,cv::Mat>(cursor,blob_visual));

			centroids_.insert(std::pair<int,cv::Point2f>(cursor,input_stream_.data_[cursor]->getCentroid() ));
			//store a maximum similarity of 1
			similarities_.insert(std::pair<int,std::vector<double> >( cursor , std::vector<double>(1,1.0) ));

			//store the bound
			initial_bounds_.insert(std::pair<int,cv::Rect>(cursor,input_stream_.data_[cursor]->LastRawBound()));

			Point2fVec src_points;
			src_points.push_back(centroids_.at(cursor));
			src_points.push_back(input_stream_.data_[cursor]->LastRawBound().tl());
			src_points.push_back(utility::BottomLeft(input_stream_.data_[cursor]->LastRawBound()));

			src_points_.insert(std::pair<int,std::vector<cv::Point2f> >(cursor, src_points));

		} else if(utility::changedMoreThanFactor( //the bound has changed a lot since first view, so reset all the periodicity tracking stuff
				initial_bounds_.find(cursor)->second,
				input_stream_.data_[cursor]->LastRawBound(), 1)) {
			ROS_INFO_STREAM("Blob "<<cursor<<" has changed size dramatically, periodicity analysis is resetting. Originial: "
					<<initial_bounds_.find(cursor)->second
					<<" new: "<<input_stream_.data_[cursor]->LastRawBound());
			//reset the template
			templates_.find(cursor)->second = blob_visual;

			//reset the centroid
			centroids_.find(cursor)->second = input_stream_.data_[cursor]->getCentroid();

			//clear the similarities and insert a 1 (for the first frame having maximum similarity)
			std::map<int,std::vector<double> >::iterator similarities_it = similarities_.find(cursor);
			similarities_it->second.clear();
			similarities_it->second.push_back(1);

			//reset initial bound
			initial_bounds_.find(cursor)->second = input_stream_.data_[cursor]->LastRawBound();

			//reset src points
			Point2fVec src_points;
			src_points.push_back(centroids_.at(cursor));
			src_points.push_back(input_stream_.data_[cursor]->LastRawBound().tl());
			src_points.push_back(utility::BottomLeft(input_stream_.data_[cursor]->LastRawBound()));

			Point2fVecMapIt src_points_it = src_points_.find(cursor);

			src_points_it->second.swap(src_points);

		} else {
			ROS_DEBUG_STREAM("Blob "<<cursor<<" didn't change significantly Originial: "
								<<initial_bounds_.find(cursor)->second
								<<" new: "<<input_stream_.data_[cursor]->LastRawBound());
			//the blob already exists, so update it by computing the similarity
			//scale and align the blob using affine transformation
			//we already have centers, so just need two more points. The corners would be good
			Point2fVec dst_points;

			dst_points.push_back(input_stream_.data_[cursor]->getCentroid());

			dst_points.push_back(input_stream_.data_[cursor]->LastRawBound().tl());
			dst_points.push_back(utility::BottomLeft(input_stream_.data_[cursor]->LastRawBound()));

			cv::Mat warp_mat = cv::getAffineTransform(dst_points,src_points_.at(cursor));


			cv::Mat warped_blob;
			cv::warpAffine(blob_visual,warped_blob,warp_mat,blob_visual.size());
			//draw the points on the warp_mat for testing/visualization
			ROS_DEBUG_STREAM("affine warp src 0: "<<src_points_[cursor][0]);
			ROS_DEBUG_STREAM("affine warp src 1: "<<src_points_[cursor][1]);
			ROS_DEBUG_STREAM("affine warp src 2: "<<src_points_[cursor][2]);

			ROS_DEBUG_STREAM("affine warp dst 0: "<<dst_points[0]);
			ROS_DEBUG_STREAM("affine warp dst 1: "<<dst_points[1]);
			ROS_DEBUG_STREAM("affine warp dst 2: "<<dst_points[2]);

			imshow("current blob warped",warped_blob);



		}
		//store results

		//show results - a mat with regions classified appropriately
		output = output | blob_visual;
	}
	imshow("axis",output);
	cv::waitKey(1);
}
