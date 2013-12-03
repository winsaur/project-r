/*
 * BlobDescriptorExtended.h
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

#ifndef BLOBDESCRIPTOREXTENDED_H_
#define BLOBDESCRIPTOREXTENDED_H_
#include <opencv2/opencv.hpp>
#include "BlobDescriptor.h"


class BlobDescriptorExtended : public BlobDescriptor {
	public:
	BlobDescriptorExtended(int sequence_number, int id, const ColorPair& colors,const cv::Rect& location,int depth,Contour* contour) : BlobDescriptor(sequence_number,id,colors,location,depth,contour) {}
	std::vector<double> backpack_similarities_,briefcase_similarities_;
	cv::Mat template_view_;

	cv::Rect initial_bound_;
	std::vector<cv::Point2f> src_points_;
	void setupPeriodic(const cv::Mat& template_view);
	void addFrame(const cv::Mat& frame);
	private:
	DISALLOW_COPY_AND_ASSIGN(BlobDescriptorExtended);

};

class BlobDescriptorExtendedFetcher {
public:
	DISALLOW_COPY_AND_ASSIGN(BlobDescriptorExtendedFetcher);
	BlobDescriptorExtendedFetcher(const std::string& topic);
	std::map<int,BlobDescriptorExtendedPtr > data_;
	void receiver(const dhs::blob& msg);
	std::string getTopic();
	std::vector<int> blobs_updated_;
private:
	ros::NodeHandle handle_;
	ros::Subscriber blob_subscriber_;
};

#endif /* BLOBDESCRIPTOREXTENDED_H_ */
