/*
 * BlobDescriptorExtended.cpp
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

#include "BlobDescriptorExtended.h"
#include "Utility.h"
#include "Types.h"

void BlobDescriptorExtended::setupPeriodic(const cv::Mat& template_view) {
	//store its template
	template_view_ = template_view;
	//store maximum similarity
	backpack_similarities_.clear();
	backpack_similarities_.push_back(1.);
	briefcase_similarities_.clear();
	briefcase_similarities_.push_back(1.);

	//store its current location
	initial_bound_ = LastRawBound();

	//store the points used for affine warp: center, upper left, lower left
	src_points_.clear();
	src_points_.push_back(getCentroid());
	src_points_.push_back(LastRawBound().tl());
	src_points_.push_back(utility::BottomLeft(LastRawBound()));
}
void BlobDescriptorExtended::addFrame(const cv::Mat& frame) {
	//get just the blob in the image
	cv::Mat warped_blob_cropped(frame,cv::Rect(initial_bound_));
	imshow("1current blob warped",warped_blob_cropped);

	//segment the nonsymmetric regions into hypothesis regions
	cv::Rect backpack_region(0,
							 initial_bound_.height*0.1,
							 initial_bound_.width,
							 initial_bound_.height*0.4);
	cv::Mat backpack_candidate(warped_blob_cropped,backpack_region);

	cv::Rect breifcase_region(0,
							 initial_bound_.height*0.5,
							 initial_bound_.width,
							 initial_bound_.height*0.5);
	cv::Mat briefcase_candidate(warped_blob_cropped,breifcase_region);

	imshow("2backpack candidate",backpack_candidate);
	imshow("3briefcase candidate",briefcase_candidate);

	//get contours in backpack and briefcase regions

}

BlobDescriptorExtendedFetcher::BlobDescriptorExtendedFetcher(const std::string& topic) {
	blob_subscriber_ = handle_.subscribe(topic,100,&BlobDescriptorExtendedFetcher::receiver,this);
}
std::string BlobDescriptorExtendedFetcher::getTopic() {
	return blob_subscriber_.getTopic();
}

void BlobDescriptorExtendedFetcher::receiver(const dhs::blob& msg) {
	//deserialize the message into the map
	//get colors
	ColorPair colors;
	if(msg.colors.size() >=3) {
		colors.first[0] = msg.colors[0];
		colors.first[1] = msg.colors[1];
		colors.first[2] = msg.colors[2];
	}
	if(msg.colors.size() >=6) {
		colors.second[0] = msg.colors[3];
		colors.second[1] = msg.colors[4];
		colors.second[2] = msg.colors[5];
	}

	//get position/dimensions
	cv::Rect position(msg.filtered_position[0],msg.filtered_position[1],msg.filtered_size[0],msg.filtered_size[1]);
	BlobDescriptorExtendedPtr new_blob(new BlobDescriptorExtended(msg.first_seen,msg.id,colors,position,msg.depth,NULL));
	std::pair<int,BlobDescriptorExtendedPtr> insert_val(msg.id,new_blob);
	std::pair<std::map<int,BlobDescriptorExtendedPtr>::iterator,bool > returned = data_.insert(insert_val);
	//if returned.second then we're done, otherwise need to update the existing blob
	if(!returned.second) {
		returned.first->second->history_.push_back(HistoryDescriptor(msg.last_seen,-1,position));
		returned.first->second->colors_ = colors;
	}

	//deserialize contour
	utility::deSerializeContour(msg.contour,returned.first->second->contour_);

	//get centroid
	returned.first->second->centroid_ = cv::Point2f(msg.centroid[0],msg.centroid[1]);

	//add this id to the blobs_updated array so something else can go do processing on it
	blobs_updated_.push_back(msg.id);
}
