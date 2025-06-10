// yolov8_utils.h
#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <vector>
#include <string>

// --- Pre & Post Processing ---

// Pre-processes a frame for the model
at::Tensor frame_process(const cv::Mat& frame, const cv::Size& input_shape);

// DFL (Distribution Focal Loss) layer implementation
at::Tensor dfl(const at::Tensor& x);

// Converts distance predictions to bounding box coordinates
at::Tensor dist2bbox(const at::Tensor& distance, const at::Tensor& anchor_points, bool xywh = true, int dim = 1);

// Performs Non-Maximum Suppression
at::Tensor non_max_suppression(const at::Tensor& prediction, float conf_thres = 0.25f, float iou_thres = 0.7f, int max_det = 300);

// --- Drawing Helpers ---

// Draws detected bounding boxes on the frame
int draw_detections(cv::Mat& frame, const at::Tensor& detections, const std::vector<std::string>& names, const std::vector<cv::Scalar>& colors);

// Draws FPS information boxes
void draw_fps_info(cv::Mat& frame, double display_fps, double detection_fps);

// Draws detection count information
void draw_detection_info(cv::Mat& frame, int num_detections);

// --- Camera Setup ---
cv::VideoCapture setup_camera();