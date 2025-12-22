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
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>   // setpriority
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
  : Node("lcd_node"), spi_fd_(-1), chip_(nullptr), line_dc_(nullptr), line_rst_(nullptr)
  {
    // Params
    declare_parameter<int>("fps", 30);
    declare_parameter<bool>("loop", true);
    declare_parameter<bool>("show_test", false);
    declare_parameter<bool>("rotate_cw_90", true);

    // SPI
    declare_parameter<int>("spi_hz", 20000000);      // istersen 8_000_000 yap
    declare_parameter<int>("spi_chunk", 4096);       // hızlı için 4096

    // Topics
    declare_parameter<std::string>("topic_media_path", "/lcd/media_path");
    declare_parameter<std::string>("topic_image", "/lcd/image");

    fps_  = std::max<int>(1, (int)get_parameter("fps").as_int());
    loop_ = get_parameter("loop").as_bool();
    show_test_ = get_parameter("show_test").as_bool();
    rotate_cw_90_ = get_parameter("rotate_cw_90").as_bool();
    spi_hz_ = std::max<int>(1000000, (int)get_parameter("spi_hz").as_int());
    spi_chunk_ = std::max<int>(256, (int)get_parameter("spi_chunk").as_int());

    topic_media_path_ = get_parameter("topic_media_path").as_string();
    topic_image_      = get_parameter("topic_image").as_string();

    RCLCPP_INFO(get_logger(), "LCD node starting...");

    if (!open_spi()) throw std::runtime_error("SPI open failed");
    if (!init_gpio()) throw std::runtime_error("GPIO init failed");

    hw_reset();
    init_lcd();

    // Subscriptions (node açılınca dinlemeye başlar)
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

    RCLCPP_INFO(get_logger(), "Listening: %s (String path) | %s (sensor_msgs/Image)",
                topic_media_path_.c_str(), topic_image_.c_str());

    if (show_test_) {
      RCLCPP_INFO(get_logger(), "show_test=true -> running test colors. Publish to topics to override.");
      timer_ = create_wall_timer(1000ms, [this]{ test_color_tick(); });
      fill_screen(0xF800);
    } else {
      fill_screen(0x0000); // black
      RCLCPP_INFO(get_logger(), "Ready. Publish media_path or Image to start.");
    }
  }

  ~LcdNode() override {
    stop_current_playback();
    if (line_dc_)  gpiod_line_release(line_dc_);
    if (line_rst_) gpiod_line_release(line_rst_);
    if (chip_)     gpiod_chip_close(chip_);
    if (spi_fd_ >= 0) close(spi_fd_);
  }

private:
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

    RCLCPP_INFO(get_logger(), "SPI opened %s mode=0 speed=%uHz bits=%u chunk=%d",
                SPI_DEV, speed, bits, spi_chunk_);
    return true;
  }

  bool spi_write_chunked(const uint8_t *data, size_t len) {
    size_t off = 0;
    while (off < len) {
      size_t n = std::min((size_t)spi_chunk_, len - off);

      struct spi_ioc_transfer tr{};
      tr.tx_buf = reinterpret_cast<unsigned long>(data + off);
      tr.rx_buf = 0;
      tr.len = n;
      tr.delay_usecs = 0;
      tr.speed_hz = 0;       // configured speed
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
    spi_write_chunked(&cmd, 1);
  }
  void write_data(uint8_t data) {
    set_dc(1);
    spi_write_chunked(&data, 1);
  }
  void write_data_buf(const uint8_t *data, size_t len) {
    set_dc(1);
    spi_write_chunked(data, len);
  }

  void init_lcd() {
    write_cmd(0x01); usleep(150000); // SWRESET
    write_cmd(0x11); usleep(150000); // SLPOUT

    write_cmd(0x3A); write_data(0x55); // RGB565

    write_cmd(0x36); write_data(0x08); // MADCTL BGR (sende doğru)
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
    std::vector<uint8_t> line(LCD_WIDTH*2);
    for (int i=0;i<LCD_WIDTH;i++){
      line[2*i]   = (color>>8)&0xFF;
      line[2*i+1] = color&0xFF;
    }
    for (int y=0;y<LCD_HEIGHT;y++){
      write_data_buf(line.data(), line.size());
    }
  }

  // ---------------- Playback control ----------------
  void stop_current_playback() {
    // token artır -> çalışan thread kendi kendine durur
    play_token_.fetch_add(1);
    running_.store(false);
    if (media_thread_.joinable()) media_thread_.join();
  }

  // ---------------- Media helpers ----------------
  bool is_image_path(const std::string& p) {
    return ends_with_ci(p, ".png") || ends_with_ci(p, ".jpg") ||
           ends_with_ci(p, ".jpeg") || ends_with_ci(p, ".bmp");
  }

  // Topic: String path
  void on_media_path(const std::string& path) {
    if (path.empty()) return;

    {
      std::lock_guard<std::mutex> lk(src_mtx_);
      current_path_ = path;
      current_is_path_ = true;
    }

    // eskiyi kes
    stop_current_playback();

    // yeni token ile başlat
    const uint64_t my_token = play_token_.load();
    running_.store(true);

    // LCD işi sesin önüne geçmesin diye (ama hızlı da kalsın)
    setpriority(PRIO_PROCESS, 0, 5);

    if (is_image_path(path)) {
      RCLCPP_INFO(get_logger(), "NEW PATH (image): %s", path.c_str());
      cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
      if (img.empty()) {
        RCLCPP_ERROR(get_logger(), "Failed to read image: %s", path.c_str());
        return;
      }
      render_bgr_frame(img);
      return;
    }

    RCLCPP_INFO(get_logger(), "NEW PATH (video): %s fps=%d loop=%d rotate_cw_90=%d",
                path.c_str(), fps_, (int)loop_, (int)rotate_cw_90_);

    media_thread_ = std::thread([this, path, my_token]{
      video_loop(path, my_token);
    });
  }

  // Topic: sensor_msgs/Image
  void on_image(const sensor_msgs::msg::Image& msg) {
    // eski video vs varsa kes
    stop_current_playback();

    // token güncel
    play_token_.fetch_add(1);

    cv::Mat bgr;
    if (!image_msg_to_bgr(msg, bgr)) {
      RCLCPP_ERROR(get_logger(), "Image msg convert failed (encoding=%s)", msg.encoding.c_str());
      return;
    }

    RCLCPP_INFO(get_logger(), "NEW IMAGE MSG: %dx%d encoding=%s",
                msg.width, msg.height, msg.encoding.c_str());

    render_bgr_frame(bgr);
  }

  bool image_msg_to_bgr(const sensor_msgs::msg::Image& msg, cv::Mat& out_bgr) {
    // Beklenenler: "bgr8" veya "rgb8"
    const std::string enc = to_lower(msg.encoding);
    if (enc != "bgr8" && enc != "rgb8") {
      return false;
    }

    if (msg.step == 0 || msg.data.empty()) return false;

    // wrap
    cv::Mat m((int)msg.height, (int)msg.width, CV_8UC3,
              const_cast<unsigned char*>(msg.data.data()),
              (size_t)msg.step);

    if (enc == "bgr8") {
      out_bgr = m.clone();
      return true;
    }

    // rgb8 -> bgr
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
    while (running_.load()) {
      // yeni bir şey geldiyse kendini bırak
      if (play_token_.load() != token) break;

      if (!cap.read(frame) || frame.empty()) {
        if (loop_) {
          cap.set(cv::CAP_PROP_POS_FRAMES, 0);
          continue;
        }
        break;
      }

      render_bgr_frame(frame);

      next += period;
      std::this_thread::sleep_until(next);
    }
  }

  // ---- FAST render: full-window once, then stream lines ----
  void render_bgr_frame(const cv::Mat& bgr) {
    cv::Mat rotated;
    if (rotate_cw_90_) {
      cv::rotate(bgr, rotated, cv::ROTATE_90_CLOCKWISE);
    } else {
      rotated = bgr;
    }

    cv::Mat resized;
    cv::resize(rotated, resized, cv::Size(LCD_WIDTH, LCD_HEIGHT), 0, 0, cv::INTER_LINEAR);

    // window full screen ONCE (hız)
    set_addr_window(0, 0, LCD_WIDTH - 1, LCD_HEIGHT - 1);

    std::vector<uint8_t> line((size_t)LCD_WIDTH * 2);

    for (int y = 0; y < LCD_HEIGHT; ++y) {
      const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);

      size_t idx = 0;
      for (int x = 0; x < LCD_WIDTH; ++x) {
        uint8_t B = row[x][0];
        uint8_t G = row[x][1];
        uint8_t R = row[x][2];

        uint16_t rgb565 =
          ((R & 0xF8) << 8) |
          ((G & 0xFC) << 3) |
          ((B & 0xF8) >> 3);

        line[idx++] = (rgb565 >> 8) & 0xFF;
        line[idx++] = rgb565 & 0xFF;
      }

      write_data_buf(line.data(), line.size());
    }
  }

  // ---------- Test ----------
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
  int spi_chunk_{4096};

  std::string topic_media_path_{"/lcd/media_path"};
  std::string topic_image_{"/lcd/image"};

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
  std::atomic<bool> running_{false};
  std::thread media_thread_;
  std::atomic<uint64_t> play_token_{1};

  std::mutex src_mtx_;
  std::string current_path_;
  bool current_is_path_{true};

  int test_idx_{0};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LcdNode>());
  rclcpp::shutdown();
  return 0;
}