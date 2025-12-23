#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>
#include <gpiod.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <errno.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"

using namespace std::chrono_literals;

static constexpr const char * SPI_DEV = "/dev/spidev0.0";

// Waveshare 2inch TFT (ST7789V) pinout (BCM)
static constexpr unsigned int PIN_DC  = 25;
static constexpr unsigned int PIN_RST = 27;

// LCD resolution
static constexpr int LCD_WIDTH  = 240;
static constexpr int LCD_HEIGHT = 320;

static inline std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return (char)std::tolower(c); });
  return s;
}

static inline bool ends_with_ci(const std::string& s, const std::string& suffix) {
  if (suffix.size() > s.size()) return false;
  std::string a = to_lower(s.substr(s.size() - suffix.size()));
  std::string b = to_lower(suffix);
  return a == b;
}

class LcdNode : public rclcpp::Node {
public:
  LcdNode()
  : Node("lcd_node"),
    spi_fd_(-1),
    chip_(nullptr),
    line_dc_(nullptr),
    line_rst_(nullptr)
  {
    // ---------- Core params ----------
    declare_parameter<int>("fps", 30);
    declare_parameter<bool>("loop", true);
    declare_parameter<bool>("show_test", false);
    declare_parameter<bool>("rotate_cw_90", true);

    // ---------- SPI params ----------
    declare_parameter<int>("spi_hz", 20000000);
    declare_parameter<int>("spi_chunk", 32768);          // ✅ bigger default
    declare_parameter<bool>("spi_use_multi_ioc", false); // optional advanced mode

    // ---------- Topics ----------
    declare_parameter<std::string>("topic_media_path", "/lcd/media_path");
    declare_parameter<std::string>("topic_image", "/lcd/image");

    // ---------- ROI params ----------
    declare_parameter<bool>("roi_enable", true);
    declare_parameter<int>("roi_pad", 2);

    // ---------- Apply params ----------
    reload_params();

    RCLCPP_INFO(get_logger(), "LCD node starting...");

    if (!open_spi()) throw std::runtime_error("SPI open failed");
    if (!init_gpio()) throw std::runtime_error("GPIO init failed");

    hw_reset();
    init_lcd();

    // ---------- Subscriptions ----------
    sub_path_ = create_subscription<std_msgs::msg::String>(
      topic_media_path_, 10,
      [this](std_msgs::msg::String::SharedPtr msg){
        on_media_path(msg->data);
      }
    );

    sub_img_ = create_subscription<sensor_msgs::msg::Image>(
      topic_image_, 10,
      [this](sensor_msgs::msg::Image::SharedPtr msg){
        on_image(*msg);
      }
    );

    // ---------- SPI worker thread ----------
    running_.store(true);
    spi_worker_ = std::thread([this]{ spi_worker_loop(); });

    // ---------- test ----------
    if (show_test_) {
      RCLCPP_INFO(get_logger(), "show_test=true -> running test colors. Publish to topics to override.");
      timer_ = create_wall_timer(1000ms, [this]{ test_color_tick(); });
      fill_screen(0xF800);
    } else {
      fill_screen(0x0000);
      RCLCPP_INFO(get_logger(), "Ready. Publish /lcd/media_path or /lcd/image to start.");
    }

    RCLCPP_INFO(get_logger(),
      "Listening: %s (String path) | %s (sensor_msgs/Image) | ROI=%d pad=%d | fps=%d | spi_hz=%d | chunk=%d | multi_ioc=%d",
      topic_media_path_.c_str(),
      topic_image_.c_str(),
      (int)roi_enable_, roi_pad_,
      fps_, spi_hz_, spi_chunk_, (int)spi_use_multi_ioc_
    );
  }

  ~LcdNode() override {
    stop_current_playback();

    running_.store(false);
    {
      std::lock_guard<std::mutex> lk(frame_mtx_);
      frame_ready_ = true;
    }
    frame_cv_.notify_all();
    if (spi_worker_.joinable()) spi_worker_.join();

    if (line_dc_)  gpiod_line_release(line_dc_);
    if (line_rst_) gpiod_line_release(line_rst_);
    if (chip_)     gpiod_chip_close(chip_);
    if (spi_fd_ >= 0) close(spi_fd_);
  }

private:
  // ---------- params reload ----------
  void reload_params() {
    fps_  = std::max<int>(1, (int)get_parameter("fps").as_int());
    loop_ = get_parameter("loop").as_bool();
    show_test_ = get_parameter("show_test").as_bool();
    rotate_cw_90_ = get_parameter("rotate_cw_90").as_bool();

    spi_hz_ = std::max<int>(1000000, (int)get_parameter("spi_hz").as_int());
    spi_chunk_ = std::max<int>(256, (int)get_parameter("spi_chunk").as_int());
    spi_use_multi_ioc_ = get_parameter("spi_use_multi_ioc").as_bool();

    topic_media_path_ = get_parameter("topic_media_path").as_string();
    topic_image_      = get_parameter("topic_image").as_string();

    roi_enable_ = get_parameter("roi_enable").as_bool();
    roi_pad_    = std::max<int>(0, (int)get_parameter("roi_pad").as_int());
  }

  // ---------------- SPI ----------------
  bool open_spi() {
    spi_fd_ = ::open(SPI_DEV, O_RDWR);
    if (spi_fd_ < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to open %s", SPI_DEV);
      return false;
    }

    uint8_t mode = SPI_MODE_0;
    uint32_t speed = (uint32_t)spi_hz_;
    uint8_t bits = 8;

    if (ioctl(spi_fd_, SPI_IOC_WR_MODE, &mode) < 0) {
      RCLCPP_ERROR(get_logger(), "SPI set mode failed");
      return false;
    }
    if (ioctl(spi_fd_, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
      RCLCPP_ERROR(get_logger(), "SPI set speed failed");
      return false;
    }
    if (ioctl(spi_fd_, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0) {
      RCLCPP_ERROR(get_logger(), "SPI set bits-per-word failed");
      return false;
    }

    RCLCPP_INFO(get_logger(), "SPI opened %s mode=0 speed=%uHz bits=%u chunk=%d multi_ioc=%d",
                SPI_DEV, speed, bits, spi_chunk_, (int)spi_use_multi_ioc_);
    return true;
  }

  bool spi_write_chunked_single_ioc(const uint8_t *data, size_t len) {
    // Old style but with bigger chunk (still ok)
    size_t off = 0;
    while (off < len) {
      size_t n = std::min((size_t)spi_chunk_, len - off);

      struct spi_ioc_transfer tr{};
      tr.tx_buf = reinterpret_cast<unsigned long>(data + off);
      tr.rx_buf = 0;
      tr.len = n;
      tr.delay_usecs = 0;
      tr.speed_hz = 0;
      tr.bits_per_word = 8;

      int ret = ioctl(spi_fd_, SPI_IOC_MESSAGE(1), &tr);
      if (ret < 0) {
        RCLCPP_ERROR(get_logger(), "SPI write failed (len=%zu) errno=%d", n, errno);
        return false;
      }
      off += n;
    }
    return true;
  }

  bool spi_write_chunked_multi_ioc(const uint8_t *data, size_t len) {
    // One ioctl with many transfers (reduces syscalls further)
    // We cap transfers count to avoid huge stack; allocate vector.
    size_t off = 0;
    std::vector<spi_ioc_transfer> trs;
    trs.reserve((len / spi_chunk_) + 2);

    while (off < len) {
      size_t n = std::min((size_t)spi_chunk_, len - off);
      spi_ioc_transfer tr{};
      tr.tx_buf = reinterpret_cast<unsigned long>(data + off);
      tr.rx_buf = 0;
      tr.len = n;
      tr.delay_usecs = 0;
      tr.speed_hz = 0;
      tr.bits_per_word = 8;
      trs.push_back(tr);
      off += n;
    }

    int ret = ioctl(spi_fd_, SPI_IOC_MESSAGE((int)trs.size()), trs.data());
    if (ret < 0) {
      RCLCPP_ERROR(get_logger(), "SPI write failed (multi_ioc transfers=%zu) errno=%d", trs.size(), errno);
      return false;
    }
    return true;
  }

  bool spi_write(const uint8_t *data, size_t len) {
    if (spi_use_multi_ioc_) return spi_write_chunked_multi_ioc(data, len);
    return spi_write_chunked_single_ioc(data, len);
  }

  // ---------------- GPIO ----------------
  bool init_gpio() {
    chip_ = gpiod_chip_open_by_name("gpiochip4");
    if (!chip_) {
      RCLCPP_ERROR(get_logger(), "Failed to open gpiochip4 (permission?)");
      return false;
    }
    RCLCPP_INFO(get_logger(), "LCD using /dev/gpiochip4");

    line_dc_  = gpiod_chip_get_line(chip_, PIN_DC);
    line_rst_ = gpiod_chip_get_line(chip_, PIN_RST);
    if (!line_dc_ || !line_rst_) {
      RCLCPP_ERROR(get_logger(), "Failed to get lines DC=%u RST=%u", PIN_DC, PIN_RST);
      return false;
    }

    if (gpiod_line_request_output(line_dc_, "lcd_node", 0) < 0 ||
        gpiod_line_request_output(line_rst_, "lcd_node", 1) < 0)
    {
      RCLCPP_ERROR(get_logger(), "Failed to request GPIO outputs (DC/RST)");
      return false;
    }
    return true;
  }

  void set_dc(int v) { gpiod_line_set_value(line_dc_, v); }

  void hw_reset() {
    gpiod_line_set_value(line_rst_, 0); usleep(200000);
    gpiod_line_set_value(line_rst_, 1); usleep(200000);
  }

  // ---------------- LCD low-level ----------------
  void write_cmd(uint8_t cmd) {
    set_dc(0);
    spi_write(&cmd, 1);
  }
  void write_data(uint8_t data) {
    set_dc(1);
    spi_write(&data, 1);
  }
  void write_data_buf(const uint8_t *data, size_t len) {
    set_dc(1);
    spi_write(data, len);
  }

  void init_lcd() {
    write_cmd(0x01); usleep(150000); // SWRESET
    write_cmd(0x11); usleep(150000); // SLPOUT

    write_cmd(0x3A); write_data(0x55); // RGB565

    write_cmd(0x36); write_data(0x08); // MADCTL BGR
    write_cmd(0x20);                   // INVOFF
    usleep(10000);

    write_cmd(0x13); usleep(10000);    // NORON

    // CASET
    write_cmd(0x2A);
    uint8_t caset[4] = {0x00,0x00,0x00, static_cast<uint8_t>(LCD_WIDTH-1)};
    write_data_buf(caset, 4);

    // RASET
    write_cmd(0x2B);
    uint16_t y1 = LCD_HEIGHT-1;
    uint8_t raset[4] = {0x00,0x00, static_cast<uint8_t>((y1>>8)&0xFF), static_cast<uint8_t>(y1&0xFF)};
    write_data_buf(raset, 4);

    write_cmd(0x29); usleep(120000); // DISPON
  }

  void set_addr_window(int x0, int y0, int x1, int y1) {
    uint8_t b[4];

    write_cmd(0x2A); // CASET
    b[0]=(x0>>8)&0xFF; b[1]=x0&0xFF; b[2]=(x1>>8)&0xFF; b[3]=x1&0xFF;
    write_data_buf(b,4);

    write_cmd(0x2B); // RASET
    b[0]=(y0>>8)&0xFF; b[1]=y0&0xFF; b[2]=(y1>>8)&0xFF; b[3]=y1&0xFF;
    write_data_buf(b,4);

    write_cmd(0x2C); // RAMWR
  }

  void fill_screen(uint16_t color) {
    set_addr_window(0,0,LCD_WIDTH-1,LCD_HEIGHT-1);
    std::vector<uint8_t> buf((size_t)LCD_WIDTH * (size_t)LCD_HEIGHT * 2);
    for (int i=0;i<LCD_WIDTH*LCD_HEIGHT;i++){
      buf[2*i]   = (color>>8)&0xFF;
      buf[2*i+1] = color&0xFF;
    }
    write_data_buf(buf.data(), buf.size());
  }

  // ---------------- Playback control ----------------
  void stop_current_playback() {
    play_token_.fetch_add(1);
    media_running_.store(false);
    if (media_thread_.joinable()) media_thread_.join();
  }

  // ---------------- Media helpers ----------------
  bool is_image_path(const std::string& p) {
    return ends_with_ci(p, ".png") || ends_with_ci(p, ".jpg") ||
           ends_with_ci(p, ".jpeg") || ends_with_ci(p, ".bmp");
  }

  void on_media_path(const std::string& path) {
    if (path.empty()) return;

    stop_current_playback();

    const uint64_t my_token = play_token_.load();
    media_running_.store(true);

    if (is_image_path(path)) {
      RCLCPP_INFO(get_logger(), "NEW PATH (image): %s", path.c_str());
      cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
      if (img.empty()) {
        RCLCPP_ERROR(get_logger(), "Failed to read image: %s", path.c_str());
        return;
      }
      render_bgr_frame_enqueue(img);
      return;
    }

    RCLCPP_INFO(get_logger(), "NEW PATH (video): %s fps=%d loop=%d rotate_cw_90=%d ROI=%d",
                path.c_str(), fps_, (int)loop_, (int)rotate_cw_90_, (int)roi_enable_);

    media_thread_ = std::thread([this, path, my_token]{
      video_loop(path, my_token);
    });
  }

  void on_image(const sensor_msgs::msg::Image& msg) {
    stop_current_playback();

    cv::Mat bgr;
    if (!image_msg_to_bgr(msg, bgr)) {
      RCLCPP_ERROR(get_logger(), "Image msg convert failed (encoding=%s)", msg.encoding.c_str());
      return;
    }

    RCLCPP_INFO(get_logger(), "NEW IMAGE MSG: %ux%u encoding=%s",
                msg.width, msg.height, msg.encoding.c_str());

    render_bgr_frame_enqueue(bgr);
  }

  bool image_msg_to_bgr(const sensor_msgs::msg::Image& msg, cv::Mat& out_bgr) {
    const std::string enc = to_lower(msg.encoding);
    if (enc != "bgr8" && enc != "rgb8") return false;
    if (msg.step == 0 || msg.data.empty()) return false;

    cv::Mat m((int)msg.height, (int)msg.width, CV_8UC3,
              const_cast<unsigned char*>(msg.data.data()),
              (size_t)msg.step);

    if (enc == "bgr8") {
      out_bgr = m.clone();
      return true;
    }
    cv::cvtColor(m, out_bgr, cv::COLOR_RGB2BGR);
    return true;
  }

  void video_loop(const std::string& path, uint64_t token) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Failed to open video: %s", path.c_str());
      return;
    }

    const auto period = std::chrono::nanoseconds((long long)(1e9 / fps_));
    auto next = std::chrono::steady_clock::now();

    cv::Mat frame;
    while (media_running_.load()) {
      if (play_token_.load() != token) break;

      if (!cap.read(frame) || frame.empty()) {
        if (loop_) {
          cap.set(cv::CAP_PROP_POS_FRAMES, 0);
          continue;
        }
        break;
      }

      render_bgr_frame_enqueue(frame);

      next += period;
      std::this_thread::sleep_until(next);
    }
  }

  // ---------------- Producer: frame->rgb565 + ROI compute -> enqueue ----------------
  void render_bgr_frame_enqueue(const cv::Mat& bgr) {
    // rotate + resize
    cv::Mat rotated;
    if (rotate_cw_90_) cv::rotate(bgr, rotated, cv::ROTATE_90_CLOCKWISE);
    else rotated = bgr;

    cv::Mat resized;
    cv::resize(rotated, resized, cv::Size(LCD_WIDTH, LCD_HEIGHT), 0, 0, cv::INTER_LINEAR);

    // cur565 as uint16
    std::vector<uint16_t> cur565((size_t)LCD_WIDTH * (size_t)LCD_HEIGHT);
    for (int y = 0; y < LCD_HEIGHT; ++y) {
      const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
      size_t rowoff = (size_t)y * LCD_WIDTH;
      for (int x = 0; x < LCD_WIDTH; ++x) {
        uint8_t B = row[x][0], G = row[x][1], R = row[x][2];
        uint16_t rgb565 =
          ((R & 0xF8) << 8) |
          ((G & 0xFC) << 3) |
          ((B & 0xF8) >> 3);
        cur565[rowoff + (size_t)x] = rgb565;
      }
    }

    int minx = 0, miny = 0, maxx = LCD_WIDTH - 1, maxy = LCD_HEIGHT - 1;

    if (roi_enable_ && prev_valid_ && prev565_.size() == cur565.size()) {
      minx = LCD_WIDTH; miny = LCD_HEIGHT; maxx = -1; maxy = -1;

      for (int y = 0; y < LCD_HEIGHT; ++y) {
        size_t rowoff = (size_t)y * LCD_WIDTH;
        for (int x = 0; x < LCD_WIDTH; ++x) {
          uint16_t a = cur565[rowoff + (size_t)x];
          uint16_t b = prev565_[rowoff + (size_t)x];
          if (a != b) {
            if (x < minx) minx = x;
            if (y < miny) miny = y;
            if (x > maxx) maxx = x;
            if (y > maxy) maxy = y;
          }
        }
      }

      if (maxx < 0) {
        prev565_.swap(cur565);
        prev_valid_ = true;
        return;
      }

      minx = std::max(0, minx - roi_pad_);
      miny = std::max(0, miny - roi_pad_);
      maxx = std::min(LCD_WIDTH - 1,  maxx + roi_pad_);
      maxy = std::min(LCD_HEIGHT - 1, maxy + roi_pad_);
    }

    // Build ONE contiguous RGB565 byte buffer for ROI
    const int roi_w = maxx - minx + 1;
    const int roi_h = maxy - miny + 1;
    std::vector<uint8_t> roi_bytes((size_t)roi_w * (size_t)roi_h * 2);

    size_t idx = 0;
    for (int y = miny; y <= maxy; ++y) {
      size_t rowoff = (size_t)y * LCD_WIDTH;
      for (int x = minx; x <= maxx; ++x) {
        uint16_t v = cur565[rowoff + (size_t)x];
        roi_bytes[idx++] = (v >> 8) & 0xFF;
        roi_bytes[idx++] = v & 0xFF;
      }
    }

    // enqueue "latest frame" (drop older)
    {
      std::lock_guard<std::mutex> lk(frame_mtx_);
      pending_.minx = minx;
      pending_.miny = miny;
      pending_.maxx = maxx;
      pending_.maxy = maxy;
      pending_.rgb565_bytes.swap(roi_bytes);
      frame_ready_ = true;
    }
    frame_cv_.notify_one();

    prev565_.swap(cur565);
    prev_valid_ = true;
  }

  // ---------------- Consumer: SPI worker only ----------------
  struct PendingFrame {
    int minx{0}, miny{0}, maxx{LCD_WIDTH-1}, maxy{LCD_HEIGHT-1};
    std::vector<uint8_t> rgb565_bytes;
  };

  void spi_worker_loop() {
    while (running_.load()) {
      PendingFrame local;

      {
        std::unique_lock<std::mutex> lk(frame_mtx_);
        frame_cv_.wait(lk, [this]{ return frame_ready_; });
        if (!running_.load()) break;

        local = std::move(pending_);
        pending_ = PendingFrame{};
        frame_ready_ = false;
      }

      if (local.rgb565_bytes.empty()) continue;

      // window once, then one big write
      set_addr_window(local.minx, local.miny, local.maxx, local.maxy);
      write_data_buf(local.rgb565_bytes.data(), local.rgb565_bytes.size());
    }
  }

  // ---------------- Test ----------------
  void test_color_tick() {
    static const uint16_t colors[] = {0xF800,0x07E0,0x001F};
    test_idx_ = (test_idx_ + 1) % 3;
    fill_screen(colors[test_idx_]);
  }

private:
  // params
  int fps_{30};
  bool loop_{true};
  bool show_test_{false};
  bool rotate_cw_90_{true};

  int spi_hz_{20000000};
  int spi_chunk_{32768};
  bool spi_use_multi_ioc_{false};

  std::string topic_media_path_{"/lcd/media_path"};
  std::string topic_image_{"/lcd/image"};

  bool roi_enable_{true};
  int roi_pad_{2};

  // hw
  int spi_fd_;
  gpiod_chip *chip_;
  gpiod_line *line_dc_;
  gpiod_line *line_rst_;

  // ros
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_path_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
  rclcpp::TimerBase::SharedPtr timer_;

  // playback
  std::atomic<bool> media_running_{false};
  std::thread media_thread_;
  std::atomic<uint64_t> play_token_{1};

  // SPI worker
  std::atomic<bool> running_{false};
  std::thread spi_worker_;

  std::mutex frame_mtx_;
  std::condition_variable frame_cv_;
  PendingFrame pending_;
  bool frame_ready_{false};

  // ROI history
  std::vector<uint16_t> prev565_;
  bool prev_valid_{false};

  int test_idx_{0};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LcdNode>());
  rclcpp::shutdown();
  return 0;
}