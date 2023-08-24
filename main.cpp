/* face tracking PC application SDK */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include "detection/face_detector.h"
#include "face_tracker.h"

#include <string>
#include <sstream>

using std::vector;
using namespace cv;
using namespace std;

typedef struct {
	std::vector<int> id_vector;
	std::vector<cv::Mat> img_vector;
} TrackData;

void DrawDetection(cv::Mat& img, std::vector<FaceInfo> boxes){
	if (boxes.size() == 0) {
		//cv::imshow("image", img);
	}
	else {
		for (int i = 0; i < boxes.size(); i++) {
			cv::Point topleft;
			cv::Point bottomright;
			topleft.x = boxes[i].x1;
			topleft.y = boxes[i].y1;
			bottomright.x = boxes[i].x2;
			bottomright.y = boxes[i].y2;
			std::cout << topleft.x << std::endl;
			cv::rectangle(img, topleft, bottomright, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
			cv::imshow("image", img);
		}
	}
}

TrackData GetTrackData(cv::Mat& img, std::vector<Track> tracks, TrackData& track_buf) {
	TrackData data;
	if (tracks.size() != 0) {
		if (track_buf.id_vector.size() == 0) {
			for (int i = 0; i < tracks.size(); i++) {			
				cv::Point topleft;
				cv::Point bottomright;
				topleft.x = tracks[i].x1;
				topleft.y = tracks[i].y1;
				bottomright.x = tracks[i].x2;
				bottomright.y = tracks[i].y2;

				cv::Mat croppedImg;
				int width = (int)(bottomright.x - topleft.x);
				int height = (int)(bottomright.y - topleft.y);
				if (width != 0 || height != 0) {
					if ((bottomright.x < img.cols) && (bottomright.y < img.rows) && (topleft.x > 0 && topleft.y > 0)) {
						img(cv::Rect(topleft.x, topleft.y, width, height)).copyTo(croppedImg);
						data.id_vector.push_back(tracks[i].id);
						data.img_vector.push_back(croppedImg);
					}
				}
			}
			for (int i = 0; i < data.id_vector.size(); i++) {
				track_buf.id_vector.push_back(data.id_vector[i]);
				track_buf.img_vector.push_back(data.img_vector[i]);
			}
		}
		else {
			for (int i = 0; i < tracks.size(); i++) {
				if (tracks[i].id != track_buf.id_vector[i]) {
					cv::Point topleft;
					cv::Point bottomright;
					topleft.x = tracks[i].x1;
					topleft.y = tracks[i].y1;
					bottomright.x = tracks[i].x2;
					bottomright.y = tracks[i].y2;

					cv::Mat croppedImg;
					int width = (int)(bottomright.x - topleft.x);
					int height = (int)(bottomright.y - topleft.y);
					if (width != 0 || height != 0) {
						if ((bottomright.x < img.cols) && (bottomright.y < img.rows) && (topleft.x > 0 && topleft.y > 0)) {
							img(cv::Rect(topleft.x, topleft.y, width, height)).copyTo(croppedImg);
							data.id_vector.push_back(tracks[i].id);
							data.img_vector.push_back(croppedImg);						
						}
					}
				}
			}

			track_buf.id_vector.clear();
			track_buf.img_vector.clear();
			for (int i = 0; i < data.id_vector.size(); i++) {
				track_buf.id_vector.push_back(data.id_vector[i]);
				track_buf.img_vector.push_back(data.img_vector[i]);
			}
		}		
	}
	return data;
}

void WriteToFile(TrackData& data) {
	if (data.id_vector.size() != 0) {
		for (int i = 0; i < data.id_vector.size(); i++) {
			cv::String filename = "./" + std::to_string(data.id_vector[i]) + ".jpg";
			cv::imwrite(filename, data.img_vector[i]);
		}
	}
}

void DrawTrack(cv::Mat& img, std::vector<Track> tracks) {
	if (tracks.size() == 0) {
		cv::imshow("image", img);
	}
	else {
		for (int i = 0; i < tracks.size(); i++) {
			cv::Point topleft;
			cv::Point bottomright;
			topleft.x = tracks[i].x1;
			topleft.y = tracks[i].y1;
			bottomright.x = tracks[i].x2;
			bottomright.y = tracks[i].y2;
			
			cv::rectangle(img, topleft, bottomright, cv::Scalar(255, 255, 0), 2, cv::LINE_8);
			cv::putText(img, std::to_string(tracks[i].id), topleft,
				cv::FONT_HERSHEY_PLAIN, 2.0,
				cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
			cv::imshow("image", img);	
		}
	}
}

void SaveTrack(cv::Mat& img, std::vector<Track> tracks, cv::VideoWriter& oVideoWriter) {
	if (tracks.size() == 0) {
		oVideoWriter.write(img);
	}
	else {
		for (int i = 0; i < tracks.size(); i++) {
			cv::Point topleft;
			cv::Point bottomright;
			topleft.x = tracks[i].x1;
			topleft.y = tracks[i].y1;
			bottomright.x = tracks[i].x2;
			bottomright.y = tracks[i].y2;
			std::cout << topleft.x << std::endl;
			cv::rectangle(img, topleft, bottomright, cv::Scalar(255, 255, 0), 2, cv::LINE_8);
			cv::putText(img, std::to_string(tracks[i].id), topleft,
				cv::FONT_HERSHEY_PLAIN, 2.0,
				cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
			oVideoWriter.write(img);
		}
	}
}

void WebCamInference(int total_frame) {
	//Open the first camera
	cv::VideoCapture cap(0);
	//Determine if the camera is turned on
	if (!cap.isOpened())
	{
		std::cout << "Camera did not turn on successfully" << std::endl;
	}

	UltraFace detector("./weights/detection/RFB-320.bin", "./weights/detection/RFB-320.param", 320, 240, 1, 0.7);;

	FaceTracker tracker;
	int __ = tracker.LoadThirdPartyModels();
	int frame_cnt = 0;
	TrackData track_buf;
	
	while (1) {
		cv::Mat frame;
		std::vector<FaceInfo> boxes;

		bool res = cap.read(frame);
		if (!res) {
			break;
		}

		//determine whether to read
		if (frame.empty()) {
			break;
		}
		ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
		frame_cnt++;
		detector.detect(inmat, boxes);
		DrawDetection(frame, boxes);

		std::vector<Track> tracks;
		
		tracker.Get_Track(frame, boxes, tracks);
		DrawTrack(frame, tracks);
		std::cout << "\nbox: " << boxes.size() << ", tracks: " << tracks.size() << std::endl;
		if (boxes.size() == tracks.size()) {
			TrackData track_data = GetTrackData(frame, tracks, track_buf);
			WriteToFile(track_data);
		}

		if (frame_cnt == total_frame)
			break;
		//wait 1ms, exit loop if key is pressed
		if (cv::waitKey(1) >= 0)
		{
			break;
		}
	}
}


int main(int argc, char** argv){	
	int total_frame = 1000;
	std::cout << "**************************************************" << std::endl;
	std::cout << "Have " << argc << " arguments:" << std::endl;
	for (int i = 0; i < argc; ++i)
		std::cout << i << "  " << argv[i] << std::endl;
	std::cout << "**************************************************\n" << std::endl;
	for (int i = 0; i < argc; i++) {
		if (!strcmp(argv[i], "--frame_num"))
			total_frame = std::atoi(argv[i + 1]);
	}
	

	WebCamInference(total_frame);

	return 0;
}
