/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <glog/logging.h>
#include <opencv2/imgproc/types_c.h>
#include <signal.h>

#include <cassert>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include <stack>
#include <thread>
#include <type_traits>
#include "vitis/ai/bounded_queue.hpp"
#include "vitis/ai/env_config.hpp"
#include "yolov8_onnx.hpp"

DEF_ENV_PARAM(DEBUG_DEMO, "0")

// onnx_param
extern int onnx_x;
extern int onnx_y;
extern bool onnx_disable_spinning;
extern bool enable_result_print;
extern bool onnx_disable_spinning_between_run;
extern std::string intra_op_thread_affinities;

// camera and display setting
string set_cap_resolution = "";
extern string set_display_resolution = "";
int cap_width = 1920;
int cap_height = 1080;
int display_width = 1920;
int display_height = 1080;

static std::vector<std::string> split(const std::string &s,
                                      const std::string &delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}

inline std::string ToUTF8String(const std::string& s) { return s; }
std::string ToUTF8String(const std::wstring& s);

namespace vitis {
namespace ai {

// A struct for passing frames to the detection pipeline
struct FrameInfo {
  int channel_id;
  unsigned long frame_id;
  cv::Mat mat;
};

// A struct for passing detection results to the GUI
struct DetectionResult {
  unsigned long frame_id;
  Yolov8OnnxResult yolo_result;
  // The dimensions of the frame that the yolo_result corresponds to.
  int original_width = 0;
  int original_height = 0;
  // FPS Stats
  float fps = 0.0f;
  float max_fps = 0.0f;
  float min_fps = 0.0f;
  float avg_fps = 0.0f;
  std::string time_str;
};

// Define queue types for clarity
using frame_queue_t = vitis::ai::BoundedQueue<FrameInfo>;
using result_queue_t = vitis::ai::BoundedQueue<DetectionResult>;
using display_queue_t = vitis::ai::BoundedQueue<cv::Mat>;

struct MyThread {
  static inline std::vector<MyThread*>& all_threads() {
    static std::vector<MyThread*> threads;
    return threads;
  };
  static void signal_handler(int) { stop_all(); }
  static void stop_all() {
    for (auto& th : all_threads()) {
      th->stop();
    }
  }
  static void wait_all() {
    for (auto& th : all_threads()) {
      th->wait();
    }
  }
  static void start_all() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "Thread num " << all_threads().size();
    for (auto& th : all_threads()) {
      th->start();
    }
  }

  static void main_proxy(MyThread* me) { return me->main(); }
  void main() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is started";
    while (!stop_) {
      auto run_ret = run();
      if (!stop_) {
        stop_ = run_ret != 0;
      }
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "thread [" << name() << "] is ended";
  }

  virtual int run() = 0;
  virtual std::string name() = 0;

  explicit MyThread() : stop_(false), thread_{nullptr} {
    all_threads().push_back(this);
  }

  virtual ~MyThread() {
    all_threads().erase(
        std::remove(all_threads().begin(), all_threads().end(), this),
        all_threads().end());
  }

  void start() {
    thread_ = std::unique_ptr<std::thread>(new std::thread(main_proxy, this));
  }

  void stop() {
    stop_ = true;
  }

  void wait() {
    if (thread_ && thread_->joinable()) {
      thread_->join();
    }
  }
  bool is_stopped() { return stop_; }

  bool stop_;
  std::unique_ptr<std::thread> thread_;
};

struct DecodeThread : public MyThread {
  DecodeThread(int channel_id, const std::string& video_file,
               frame_queue_t* queue, display_queue_t* display_queue)
      : MyThread{},
        channel_id_{channel_id},
        video_file_{video_file},
        frame_id_{0},
        video_stream_{},
        queue_{queue},
        display_queue_{display_queue} {
    open_stream();
    auto& cap = *video_stream_.get();
    if (is_camera_) {
      cap.set(cv::CAP_PROP_FRAME_WIDTH, cap_width);
      cap.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height);
    }
  }

  virtual int run() override {
    auto& cap = *video_stream_.get();
    cv::Mat image;
    cap >> image;
    if (image.empty()) {
      open_stream();
      return 0;
    }

    // Push to display queue for playback.
    display_queue_->push(image, std::chrono::milliseconds(5));

    // Push to detection queue.
    queue_->push(FrameInfo{channel_id_, ++frame_id_, image.clone()},
                 std::chrono::milliseconds(5));

    return 0;
  }

  virtual std::string name() override {
    return std::string{"DedodeThread-"} + std::to_string(channel_id_);
  }

  void open_stream() {
    is_camera_ = video_file_.size() == 1 && video_file_[0] >= '0' &&
                 video_file_[0] <= '9';
    video_stream_ = std::unique_ptr<cv::VideoCapture>(
        is_camera_ ? new cv::VideoCapture(std::stoi(video_file_))
                   : new cv::VideoCapture(video_file_));
    if (!video_stream_->isOpened()) {
      LOG(FATAL) << "Cannot open video stream: " << video_file_;
      stop();
    }
  }

  int channel_id_;
  std::string video_file_;
  unsigned long frame_id_;
  std::unique_ptr<cv::VideoCapture> video_stream_;
  frame_queue_t* queue_;
  display_queue_t* display_queue_;
  bool is_camera_;
};

using DpuProcessResult = std::function<cv::Mat(cv::Mat&, const DetectionResult&, bool)>;

struct GuiThread : public MyThread {
  GuiThread(display_queue_t* display_queue, result_queue_t* result_queue, DpuProcessResult processor)
      : MyThread{},
        display_queue_{display_queue},
        result_queue_{result_queue},
        processor_{processor} {}

  virtual int run() override {
    cv::Mat frame;
    if (!display_queue_->pop(frame, std::chrono::milliseconds(10))) {
        auto key = cv::waitKey(1);
        if (key == 27) return 1;
        return 0;
    }
    
    // Ensure the display frame has a consistent size.
    if (frame.cols != display_width || frame.rows != display_height) {
        cv::resize(frame, frame, cv::Size(display_width, display_height));
    }
    
    auto now = std::chrono::steady_clock::now();
    display_points_.push_front(now);
    if(display_points_.size() > 1) {
      long duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - display_points_.back()).count();
      if (duration > 1000) { 
          display_fps_ = (float)(display_points_.size() - 1) * 1000.0f / (float)duration;
          while(std::chrono::duration_cast<std::chrono::milliseconds>(now - display_points_.back()).count() > 1000) {
              display_points_.pop_back();
          }
      }
    }

    DetectionResult result;
    while (result_queue_->pop(result, std::chrono::milliseconds(1))) {
      latest_result_ = result;
    }

    // Only draw if we have received at least one valid result.
    if (latest_result_.original_width > 0) {
      // The processor_ is the process_result function from demo_yolov8_onnx_n.cpp
      processor_(frame, latest_result_, false);
    }
    
    draw_helpers(frame);

    cv::imshow("YOLOv8 Decoupled", frame);
    auto key = cv::waitKey(1);
    if (key == 27) {
      return 1; // stop
    }
    return 0;
  }

  void draw_helpers(cv::Mat& frame) {
    int x = 13;
    int y = 28;
    
    std::string display_fps_str = "DISPLAY FPS: " + std::to_string(display_fps_).substr(0,4);
    cv::putText(frame, display_fps_str,
                cv::Point(x, y + 115), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2, 4);

    cv::putText(frame, latest_result_.time_str, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(178, 79, 0), 2, 4);
    cv::putText(frame, "DETECTION FPS: " + std::to_string(latest_result_.fps).substr(0,4),
                cv::Point(x, y + 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(178, 79, 0), 2, 4);
    cv::putText(frame, "Avg: " + std::to_string(latest_result_.avg_fps).substr(0,4),
                cv::Point(x, y + 50), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(178, 0, 79), 2, 4);
    float display_min_fps = (latest_result_.min_fps == std::numeric_limits<float>::max()) ? 0.0f : latest_result_.min_fps;
    cv::putText(frame, "Min: " + std::to_string(display_min_fps).substr(0,4),
                cv::Point(x, y + 70), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(178, 0, 79), 2, 4);
    cv::putText(frame, "Max: " + std::to_string(latest_result_.max_fps).substr(0,4),
                cv::Point(x, y + 90), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(178, 0, 79), 2, 4);
  }

  virtual std::string name() override { return std::string{"GUIThread"}; }

  display_queue_t* display_queue_;
  result_queue_t* result_queue_;
  DpuProcessResult processor_;
  DetectionResult latest_result_;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> display_points_;
  float display_fps_ = 0.0f;
};

template<typename dpu_model_type_t>
struct DpuThread : public MyThread {
  DpuThread(std::unique_ptr<dpu_model_type_t>&& model, frame_queue_t* queue_in,
            result_queue_t* queue_out, const std::string& suffix)
      : MyThread{},
        dpu_model_{std::move(model)},
        queue_in_{queue_in},
        queue_out_{queue_out},
        suffix_{suffix} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT DPU";
  }

  virtual int run() override {
    FrameInfo frame;
    if (!queue_in_->pop(frame, std::chrono::milliseconds(500))) {
      return 0; 
    }
    
    auto yolo_result = dpu_model_->run(frame.mat);
    
    DetectionResult result_info;
    result_info.frame_id = frame.frame_id;
    result_info.yolo_result = yolo_result;
    // Store the dimensions of the frame that was processed.
    result_info.original_width = frame.mat.cols;
    result_info.original_height = frame.mat.rows;

    if (!queue_out_->push(result_info, std::chrono::milliseconds(500))) {
      if(is_stopped()) return -1;
    }
    return 0;
  }

  virtual std::string name() override { return std::string("DPU-") + suffix_; }
  std::unique_ptr<dpu_model_type_t> dpu_model_;
  frame_queue_t* queue_in_;
  result_queue_t* queue_out_;
  std::string suffix_;
};

struct SortingThread : public MyThread {
  SortingThread(result_queue_t* queue_in, result_queue_t* queue_out,
                const std::string& suffix)
      : MyThread{},
        queue_in_{queue_in},
        queue_out_{queue_out},
        suffix_{suffix},
        fps_{0.0f},
        max_fps_{0.0f},
        min_fps_{std::numeric_limits<float>::max()},
        avg_fps_{0.0f},
        fps_sum_{0.0f},
        fps_count_{0},
        start_time_{std::chrono::steady_clock::now()} {}
  
  virtual int run() override {
    DetectionResult result_info;
    if (!queue_in_->pop(result_info, std::chrono::milliseconds(500))) {
      return 0;
    }

    auto now = std::chrono::steady_clock::now();
    long duration = 0;
    points_.push_front(now);
    if (points_.size() > 1) {
      duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - points_.back()).count();
      if (duration > 2000) { 
          fps_ = (float)(points_.size()-1) * 1000.0f / (float)duration;
          while(std::chrono::duration_cast<std::chrono::milliseconds>(now - points_.back()).count() > 2000) {
              points_.pop_back();
          }
      }
    }

    if(fps_ > 0) {
        max_fps_ = std::max(max_fps_, fps_);
        min_fps_ = std::min(min_fps_, fps_);
        fps_sum_ += fps_;
        fps_count_++;
        avg_fps_ = fps_sum_ / fps_count_;
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
    int hours = elapsed / 3600;
    int minutes = (elapsed % 3600) / 60;
    int seconds = elapsed % 60;
    std::string time_str = "Time: " + std::to_string(hours) + ":" + 
                          (minutes < 10 ? "0" : "") + std::to_string(minutes) + ":" +
                          (seconds < 10 ? "0" : "") + std::to_string(seconds);
    
    result_info.fps = fps_;
    result_info.max_fps = max_fps_;
    result_info.min_fps = min_fps_;
    result_info.avg_fps = avg_fps_;
    result_info.time_str = time_str;

    if (!queue_out_->push(result_info, std::chrono::milliseconds(500))) {
      if (is_stopped()) return -1;
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"SORT-"} + suffix_; }
  result_queue_t* queue_in_;
  result_queue_t* queue_out_;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> points_;
  std::string suffix_;
  float fps_, max_fps_, min_fps_, avg_fps_, fps_sum_;
  long fps_count_;
  std::chrono::steady_clock::time_point start_time_;
};

inline void usage_video(const char* progname) {
  std::cout << progname << " [options...] \n" <<"Options:\n      -c [parallel runs]: Specifies the (max) number of runs to invoke simultaneously. Default:1.\n"
            << "      -s [input_stream] set input stream, E.g. set 0 to use default camera.\n" 
            << "      -x [intra_op_num_threads]: Sets the number of threads used to parallelize the execution within nodes, A value of 0 means ORT will pick a default. Must >=0.\n" 
            << "      -y [inter_op_num_threads]: Sets the number of threads used to parallelize the execution of the graph (across nodes), A value of 0 means ORT will pick a default. Must >=0.\n" 
            << "      -D [Disable thread spinning]: disable spinning entirely for thread owned by onnxruntime intra-op thread pool.\n"
            << "      -Z [Force thread to stop spinning between runs]: disallow thread from spinning during runs to reduce cpu usage.\n"
            << "      -T [Set intra op thread affinities]: Specify intra op thread affinity string.\n         [Example]: -T 1,2;3,4;5,6 or -T 1-2;3-4;5-6\n         Use semicolon to separate configuration between threads.\n         E.g. 1,2;3,4;5,6 specifies affinities for three threads, the first thread will be attached to the first and second logical processor.\n"
            << "      -R [Set camera resolution]: Specify the camera resolution by string.\n         [Example]: -R 1280x720\n         Default:1920x1080.\n"
            << "      -r [Set Display resolution]: Specify the display resolution by string.\n         [Example]: -r 1280x720\n         Default:1920x1080.\n"
            << "      -L Print detection log when turning on.\n"
            << "      -h: help\n"
            << std::endl;
}

static std::vector<int> g_num_of_threads;
static std::vector<std::string> g_avi_file;

inline void parse_opt(int argc, char* argv[], int start_pos = 1) {
  int opt = 0;
  optind = start_pos;
  std::vector<std::string> sp;
  std::vector<std::string> spd;
  while ((opt = getopt(argc, argv, "s:y:x:c:T:R:r:DhLZ")) != -1) {
    switch (opt) {
      case 'c': g_num_of_threads.emplace_back(std::stoi(optarg)); break;
      case 'x': onnx_x = std::stoi(optarg); break;
      case 'y': onnx_y = std::stoi(optarg); break;
      case 'D': onnx_disable_spinning = true; break;
      case 'L': enable_result_print = true; break;
      case 'Z': onnx_disable_spinning_between_run = true; break;
      case 'T': intra_op_thread_affinities = ToUTF8String(optarg); break;
      case 'h': usage_video(argv[0]); exit(1);
      case 's': g_avi_file.push_back(optarg); break;
      case 'R':
        sp = split(optarg, "x");
        cap_width = stoi(sp[0].c_str());
        cap_height = stoi(sp[1].c_str());
        break;
      case 'r':
        spd = split(optarg, "x");
        display_width = stoi(spd[0].c_str());
        display_height = stoi(spd[1].c_str());
        break;
      default: usage_video(argv[0]); exit(1);
    }
  }
  if (g_avi_file.empty()) { g_avi_file.push_back("0"); }
  if (g_num_of_threads.empty()) { g_num_of_threads.emplace_back(1); }
}

template <typename FactoryMethod>
int main_for_video_demo(int argc, char* argv[],
                        const FactoryMethod& factory_method,
                        const DpuProcessResult& process_result) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv);
  {
    auto channel_id = 0;
    auto decode_queue = std::make_unique<frame_queue_t>(5);
    auto display_queue = std::make_unique<display_queue_t>(3); 
    auto dpu_out_queue = std::make_unique<result_queue_t>(5 * g_num_of_threads[0]);
    auto gui_result_queue = std::make_unique<result_queue_t>(5 * g_num_of_threads[0]);
    
    auto decode_thread = std::make_unique<DecodeThread>(
        channel_id, g_avi_file[0], decode_queue.get(), display_queue.get());

    auto dpu_threads = std::vector<std::unique_ptr<MyThread>>{};
    auto model_prototype = factory_method();
    using dpu_model_t = typename std::remove_reference<decltype(*model_prototype)>::type;
    
    for (int i = 0; i < g_num_of_threads[0]; ++i) {
      dpu_threads.emplace_back(new DpuThread<dpu_model_t>(
          (i == 0) ? std::move(model_prototype) : factory_method(),
          decode_queue.get(),
          dpu_out_queue.get(), 
          std::to_string(i)));
    }
    
    auto sorting_thread = std::make_unique<SortingThread>(
        dpu_out_queue.get(), gui_result_queue.get(), std::to_string(0));

    auto gui_thread = std::make_unique<GuiThread>(
        display_queue.get(), gui_result_queue.get(), process_result);

    MyThread::start_all();
    MyThread::wait_all();
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}

}  // namespace ai
}  // namespace vitis