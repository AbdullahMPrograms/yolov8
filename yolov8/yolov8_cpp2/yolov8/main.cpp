// main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <fstream>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <torch/torch.h>

#include "thread_safe_queue.h"
#include "yolov8_utils.h"

// --- Frame Handling ---

class FrameReader {
public:
    FrameReader(cv::VideoCapture& cap) : cap_(cap), running_(true) {}

    void start() {
        thread_ = std::thread(&FrameReader::run, this);
    }

    void run() {
        while (running_) {
            cv::Mat frame;
            if (!cap_.read(frame)) {
                std::cerr << "FrameReader: Failed to grab frame, stopping." << std::endl;
                running_ = false;
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(lock_);
                frame_ = frame.clone();
            }
        }
        cap_.release();
        std::cout << "FrameReader thread stopped." << std::endl;
    }

    cv::Mat get_frame() {
        std::lock_guard<std::mutex> lock(lock_);
        return frame_.empty() ? cv::Mat{} : frame_.clone();
    }

    void stop() {
        running_ = false;
    }
    
    void join() {
        if(thread_.joinable()) thread_.join();
    }

    bool is_running() { return running_; }

private:
    cv::VideoCapture& cap_;
    cv::Mat frame_;
    std::mutex lock_;
    std::atomic<bool> running_;
    std::thread thread_;
};


// --- Detection Worker ---

class DetectionWorker {
public:
    DetectionWorker(
        ThreadSafeQueue<cv::Mat>& in_queue,
        ThreadSafeQueue<at::Tensor>& out_queue,
        Ort::Session& session,
        Ort::MemoryInfo& memory_info,
        const char* input_name,
        const std::vector<int64_t>& input_dims,
        at::Tensor anchors,
        at::Tensor strides)
        : in_queue_(in_queue), out_queue_(out_queue), session_(session),
          memory_info_(memory_info), input_name_(input_name),
          input_dims_(input_dims), anchors_(anchors), strides_(strides), running_(true) {}

    void start() {
        thread_ = std::thread(&DetectionWorker::run, this);
    }

    void run() {
        while (running_) {
            auto frame_opt = in_queue_.try_pop();
            if (!frame_opt) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            cv::Mat frame = *frame_opt;
            
            // Pre-process
            at::Tensor im_tensor = frame_process(frame, cv::Size(640, 640));
            at::Tensor nhwc_tensor = im_tensor.unsqueeze(0).permute({0, 2, 3, 1});

            // Run Inference
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, nhwc_tensor.data_ptr<float>(), nhwc_tensor.numel(),
                input_dims_.data(), input_dims_.size()));

            auto output_tensors = session_.Run(
                Ort::RunOptions{nullptr}, &input_name_, input_tensors.data(), 1,
                session_.GetOutputNames().data(), session_.GetOutputNames().size());

            // Convert outputs to Torch Tensors
            std::vector<at::Tensor> outputs;
            for (auto& tensor : output_tensors) {
                auto type_info = tensor.GetTensorTypeAndShapeInfo();
                auto shape = type_info.GetShape();
                at::Tensor out_tensor = torch::from_blob(tensor.GetTensorMutableData<float>(),
                                                         at::IntArrayRef(shape.data(), shape.size()));
                outputs.push_back(out_tensor.permute({0, 3, 1, 2}).clone()); // NHWC -> NCHW
            }

            // Post-process
            auto preds = post_process(outputs);
            preds = non_max_suppression(preds, 0.25f, 0.7f, 300);

            out_queue_.try_push(preds);
        }
    }

    at::Tensor post_process(const std::vector<at::Tensor>& x) {
        std::vector<at::Tensor> reshaped_tensors;
        for (const auto& t : x) {
            reshaped_tensors.push_back(t.view({t.size(0), 144, -1}));
        }
        at::Tensor concatenated = torch::cat(reshaped_tensors, 2);
        auto split_tensors = torch::split(concatenated, {64, 80}, 1);
        auto box = split_tensors[0];
        auto cls = split_tensors[1];
        
        at::Tensor dbox = dist2bbox(dfl(box), anchors_.unsqueeze(0), true, 1) * strides_;
        return torch::cat({dbox, cls.sigmoid()}, 1);
    }
    
    void stop() {
        running_ = false;
    }

    void join() {
        if(thread_.joinable()) thread_.join();
    }

private:
    ThreadSafeQueue<cv::Mat>& in_queue_;
    ThreadSafeQueue<at::Tensor>& out_queue_;
    Ort::Session& session_;
    Ort::MemoryInfo& memory_info_;
    const char* input_name_;
    std::vector<int64_t> input_dims_;
    at::Tensor anchors_;
    at::Tensor strides_;
    std::atomic<bool> running_;
    std::thread thread_;
};


// --- Main Function ---

int main() {
    // --- Configuration ---
    const bool CAP_DISPLAY_FPS = false;
    const double TARGET_FPS = 60.0;
    const int NUM_DETECTION_WORKERS = 1;

    std::cout << "Initializing a pool of " << NUM_DETECTION_WORKERS << " workers sharing one model..." << std::endl;

    // --- Setup (Assets, Camera) ---
    std::vector<std::string> names;
    std::vector<cv::Scalar> colors;
    at::Tensor anchors, strides;
    try {
        std::ifstream f("coco.names");
        std::string line;
        while (std::getline(f, line)) names.push_back(line);
        
        cv::RNG rng(42);
        for (size_t i = 0; i < names.size(); ++i) {
            colors.emplace_back(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        }

        torch::load(anchors, "anchors.pt");
        torch::load(strides, "strides.pt");
    } catch (const std::exception& e) {
        std::cerr << "Error loading assets: " << e.what() << std::endl;
        return -1;
    }

    // --- ONNX Runtime Setup ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8-Demo");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    std::ifstream config_file("./vaip_config.json");
    std::string config_str((std::istreambuf_iterator<char>(config_file)), std::istreambuf_iterator<char>());
    std::vector<const char*> provider_keys = {"config_file"};
    std::vector<const char*> provider_values = {config_str.c_str()};
    Ort::ProviderOptions provider_options;
    provider_options.UpdateOptions(provider_keys.data(), provider_values.data(), 1);

    session_options.AppendExecutionProvider("VitisAI", provider_options);

    Ort::Session session(nullptr);
    try {
        const char* model_path = "yolov8m.onnx";
        session = Ort::Session(env, model_path, session_options);
        std::cout << "Model loaded successfully." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    input_dims[0] = 1; // Set batch size to 1

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // --- Camera, Queues, Threads ---
    cv::VideoCapture cap = setup_camera();
    ThreadSafeQueue<cv::Mat> detection_in_queue(NUM_DETECTION_WORKERS);
    ThreadSafeQueue<at::Tensor> detection_out_queue(NUM_DETECTION_WORKERS);

    FrameReader frame_reader(cap);
    
    std::vector<std::unique_ptr<DetectionWorker>> detection_workers;
    for (int i = 0; i < NUM_DETECTION_WORKERS; ++i) {
        detection_workers.push_back(std::make_unique<DetectionWorker>(
            detection_in_queue, detection_out_queue, session, memory_info,
            input_name.get(), input_dims, anchors, strides
        ));
    }
    
    frame_reader.start();
    for (auto& worker : detection_workers) {
        worker->start();
    }
    std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait for threads to initialize

    // --- Main Display Loop ---
    cv2::namedWindow("YOLOv8 Real-time Detection", cv::WINDOW_NORMAL);
    cv2::resizeWindow("YOLOv8 Real-time Detection", 1280, 720);

    // --- Stats Tracking ---
    std::deque<double> display_fps_queue, detection_fps_queue;
    const int fps_queue_size = 60;
    
    at::Tensor latest_detections;
    auto prev_loop_end_time = std::chrono::high_resolution_clock::now();
    auto prev_det_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Starting real-time detection. Press 'q' to quit." << std::endl;
    
    while (true) {
        auto loop_start_time = std::chrono::high_resolution_clock::now();
        
        cv::Mat frame = frame_reader.get_frame();
        if (frame.empty()) {
            if (!frame_reader.is_running()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        detection_in_queue.try_push(frame.clone());

        auto detection_result = detection_out_queue.try_pop();
        if (detection_result) {
            latest_detections = *detection_result;
            
            auto curr_det_time = std::chrono::high_resolution_clock::now();
            double delta = std::chrono::duration<double>(curr_det_time - prev_det_time).count();
            if (delta > 0) {
                detection_fps_queue.push_back(1.0 / delta);
                if (detection_fps_queue.size() > fps_queue_size) detection_fps_queue.pop_front();
            }
            prev_det_time = curr_det_time;
        }

        cv::Mat annotated_frame = frame;
        int num_detections = latest_detections.defined() ? draw_detections(annotated_frame, latest_detections, names, colors) : 0;
        
        double avg_display_fps = display_fps_queue.empty() ? 0.0 : std::accumulate(display_fps_queue.begin(), display_fps_queue.end(), 0.0) / display_fps_queue.size();
        double avg_detection_fps = detection_fps_queue.empty() ? 0.0 : std::accumulate(detection_fps_queue.begin(), detection_fps_queue.end(), 0.0) / detection_fps_queue.size();

        draw_fps_info(annotated_frame, avg_display_fps, avg_detection_fps);
        draw_detection_info(annotated_frame, num_detections);

        cv::imshow("YOLOv8 Real-time Detection", annotated_frame);
        if (cv::waitKey(1) == 'q') break;

        auto current_loop_end_time = std::chrono::high_resolution_clock::now();
        double actual_frame_time = std::chrono::duration<double>(current_loop_end_time - prev_loop_end_time).count();
        prev_loop_end_time = current_loop_end_time;
        if (actual_frame_time > 0) {
            display_fps_queue.push_back(1.0 / actual_frame_time);
            if (display_fps_queue.size() > fps_queue_size) display_fps_queue.pop_front();
        }
    }

    // --- Cleanup ---
    std::cout << "Stopping threads..." << std::endl;
    frame_reader.stop();
    for (auto& worker : detection_workers) {
        worker->stop();
    }
    
    frame_reader.join();
    for (auto& worker : detection_workers) {
        worker->join();
    }
    
    session.release();
    allocator.Free(input_name.release());

    cv::destroyAllWindows();
    std::cout << "Application finished." << std::endl;

    return 0;
}