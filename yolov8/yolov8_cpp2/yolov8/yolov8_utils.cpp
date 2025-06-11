// yolov8_utils.cpp
#include "yolov8_utils.h"

// --- Pre & Post Processing Implementations ---

at::Tensor frame_process(const cv::Mat& frame, const cv::Size& input_shape) {
    cv::Mat rgb_frame, resized_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    cv::resize(rgb_frame, resized_frame, input_shape);

    at::Tensor tensor_image = torch::from_blob(resized_frame.data, {resized_frame.rows, resized_frame.cols, 3}, at::kByte);
    tensor_image = tensor_image.to(at::kFloat).div(255.0);
    tensor_image = tensor_image.permute({2, 0, 1}); // HWC to CHW
    return tensor_image;
}

at::Tensor dfl(const at::Tensor& x) {
    int c1 = 16;
    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, 1, 1).bias(false));
    conv->weight.set_data(torch::arange(c1, at::kFloat).view({1, c1, 1, 1}));
    conv->eval(); // Set to evaluation mode
    
    auto sizes = x.sizes();
    int64_t b = sizes[0];
    int64_t a = sizes[2];
    return conv(x.view({b, 4, c1, a}).transpose(1, 2).softmax(1)).view({b, 4, a});
}

at::Tensor dist2bbox(const at::Tensor& distance, const at::Tensor& anchor_points, bool xywh, int dim) {
    auto split_tensors = torch::split(distance, 2, dim);
    auto lt = split_tensors[0];
    auto rb = split_tensors[1];
    auto x1y1 = anchor_points - lt;
    auto x2y2 = anchor_points + rb;
    if (xywh) {
        auto c_xy = (x1y1 + x2y2) / 2;
        auto wh = x2y2 - x1y1;
        return torch::cat({c_xy, wh}, dim);
    }
    return torch::cat({x1y1, x2y2}, dim);
}

at::Tensor non_max_suppression(const at::Tensor& prediction, float conf_thres, float iou_thres, int max_det) {
    // Prediction shape: [batch_size, num_classes + 4, num_anchors] -> [1, 84, 8400]
    // Transpose to [1, 8400, 84] for easier processing
    at::Tensor pred = prediction.permute({0, 2, 1});

    std::vector<at::Tensor> output;
    for (int i = 0; i < pred.size(0); ++i) {
        at::Tensor x = pred[i]; // Shape [8400, 84]

        // Filter by confidence
        x = x.index({x.select(1, 4) > conf_thres});
        if (x.size(0) == 0) {
            output.push_back(torch::empty({0, 6}));
            continue;
        }

        // Add object confidence to class confidence
        x.slice(1, 5, x.size(1)) *= x.select(1, 4).unsqueeze(1);

        // Box (center x, center y, width, height) to (x1, y1, x2, y2)
        at::Tensor box = x.select(1, 0).unsqueeze(1); // Placeholder for box conversion
        at::Tensor cx = x.select(1, 0);
        at::Tensor cy = x.select(1, 1);
        at::Tensor w = x.select(1, 2);
        at::Tensor h = x.select(1, 3);
        at::Tensor x1 = cx - w / 2;
        at::Tensor y1 = cy - h / 2;
        at::Tensor x2 = cx + w / 2;
        at::Tensor y2 = cy + h / 2;
        box = torch::stack({x1, y1, x2, y2}, 1);

        // Get score and class index
        auto max_result = torch::max(x.slice(1, 5, x.size(1)), 1);
        at::Tensor max_conf = std::get<0>(max_result);
        at::Tensor max_idx = std::get<1>(max_result);

        x = torch::cat({box, max_conf.unsqueeze(1), max_idx.to(at::kFloat).unsqueeze(1)}, 1);
        
        // NMS
        std::vector<at::Tensor> detections;
        while(x.size(0) > 0) {
            auto order = std::get<1>(x.select(1, 4).sort(0, /*descending*/ true));
            x = x.index_select(0, order);

            at::Tensor best = x[0];
            detections.push_back(best);
            x = x.slice(0, 1, x.size(0));

            if (x.size(0) == 0) break;

            at::Tensor b1 = best.slice(0, 0, 4);
            at::Tensor b2s = x.slice(1, 0, 4);

            at::Tensor inter_x1 = torch::max(b1[0], b2s.select(1, 0));
            at::Tensor inter_y1 = torch::max(b1[1], b2s.select(1, 1));
            at::Tensor inter_x2 = torch::min(b1[2], b2s.select(1, 2));
            at::Tensor inter_y2 = torch::min(b1[3], b2s.select(1, 3));

            at::Tensor inter_w = torch::clamp(inter_x2 - inter_x1, 0);
            at::Tensor inter_h = torch::clamp(inter_y2 - inter_y1, 0);
            at::Tensor inter_area = inter_w * inter_h;

            at::Tensor b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1]);
            at::Tensor b2_area = (b2s.select(1, 2) - b2s.select(1, 0)) * (b2s.select(1, 3) - b2s.select(1, 1));
            
            at::Tensor iou = inter_area / (b1_area + b2_area - inter_area + 1e-5);
            
            x = x.index({iou < iou_thres});
        }
        if (detections.size() > 0) {
            output.push_back(torch::stack(detections));
        } else {
            output.push_back(torch::empty({0, 6}));
        }
    }
    // For this app, we only care about the first batch item
    return output[0].slice(0, 0, std::min((long)max_det, (long)output[0].size(0)));
}


// --- Drawing Helpers Implementations ---

int draw_detections(cv::Mat& frame, const at::Tensor& detections, const std::vector<std::string>& names, const std::vector<cv::Scalar>& colors) {
    if (detections.numel() == 0) {
        return 0;
    }

    float scale_x = static_cast<float>(frame.cols) / 640.0f;
    float scale_y = static_cast<float>(frame.rows) / 640.0f;

    auto det_accessor = detections.accessor<float, 2>();

    for (int i = 0; i < detections.size(0); ++i) {
        float x1 = det_accessor[i][0] * scale_x;
        float y1 = det_accessor[i][1] * scale_y;
        float x2 = det_accessor[i][2] * scale_x;
        float y2 = det_accessor[i][3] * scale_y;
        float conf = det_accessor[i][4];
        int cls_id = static_cast<int>(det_accessor[i][5]);
        
        cv::Scalar color = colors[cls_id % colors.size()];
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        std::string label = names[cls_id] + " " + cv::format("%.2f", conf);
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        cv::rectangle(frame, cv::Point(x1, y1 - label_size.height - 10), cv::Point(x1 + label_size.width, y1), color, -1);
        cv::putText(frame, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    return detections.size(0);
}

void draw_fps_info(cv::Mat& frame, double display_fps, double detection_fps) {
    auto draw_box = [&](cv::Point pos, cv::Size size, const std::string& title, double fps_val, cv::Scalar fps_color) {
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, pos, cv::Point(pos.x + size.width, pos.y + size.height), cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.6, frame, 0.4, 0, frame);
        cv::rectangle(frame, pos, cv::Point(pos.x + size.width, pos.y + size.height), cv::Scalar(100, 100, 100), 2);
        cv::putText(frame, title, cv::Point(pos.x + 10, pos.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(frame, "FPS: " + cv::format("%.1f", fps_val), cv::Point(pos.x + 10, pos.y + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1);
    };

    cv::Scalar display_fps_color = (display_fps > 50) ? cv::Scalar(0, 255, 0) : (display_fps > 30) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
    cv::Size box_size(200, 50);
    draw_box({10, 10}, box_size, "Display", display_fps, display_fps_color);
    
    cv::Scalar det_fps_color = (detection_fps > 30) ? cv::Scalar(0, 255, 0) : (detection_fps > 20) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
    draw_box({10, 10 + box_size.height + 10}, box_size, "Detection", detection_fps, det_fps_color);
}


void draw_detection_info(cv::Mat& frame, int num_detections) {
    if (num_detections <= 0) return;
    std::string text = "Detections: " + std::to_string(num_detections);
    int baseLine;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
    cv::Point pos(frame.cols - text_size.width - 20, 35);

    cv::rectangle(frame, {pos.x - 10, pos.y - text_size.height - 5}, {pos.x + text_size.width + 10, pos.y + 5}, cv::Scalar(0,0,0), -1);
    cv::rectangle(frame, {pos.x - 10, pos.y - text_size.height - 5}, {pos.x + text_size.width + 10, pos.y + 5}, cv::Scalar(100,100,100), 2);
    cv::putText(frame, text, pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
}

// --- Camera Setup Implementation ---

cv::VideoCapture setup_camera() {
    int CAMERA_INDEX = 0;
    std::cout << "Attempting to open camera index: " << CAMERA_INDEX << std::endl;
    cv::VideoCapture cap(CAMERA_INDEX, cv::CAP_DSHOW);
    
    if (!cap.isOpened()) {
        std::cout << "DSHOW backend failed. Trying default backend..." << std::endl;
        cap.open(CAMERA_INDEX);
    }
    
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open webcam. Please check camera index and permissions.");
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 60);
    
    int actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "Successfully opened camera " << CAMERA_INDEX << "." << std::endl;
    std::cout << "Camera settings: " << actual_width << "x" << actual_height << " @ " << cv::format("%.2f", actual_fps) << " FPS" << std::endl;
    
    return cap;
}