#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int16_multi_array.hpp>

#include <gpiod.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <thread>

using namespace std::chrono_literals;

static inline int clampi(int x, int lo, int hi) {
  return (x < lo) ? lo : (x > hi) ? hi : x;
}

class MotorDriverNode : public rclcpp::Node {
public:
  MotorDriverNode() : Node("motor_driver_node") {
    this->declare_parameter<int>("left_fwd", 26);
    this->declare_parameter<int>("left_rev", 13);
    this->declare_parameter<int>("right_fwd", 16);
    this->declare_parameter<int>("right_rev", 24);

    this->declare_parameter<std::string>("gpiochip", "gpiochip4");

    this->declare_parameter<int>("pwm_hz", 200);
    this->declare_parameter<int>("timeout_ms", 500);

    this->declare_parameter<bool>("invert_left", false);
    this->declare_parameter<bool>("invert_right", false);

    left_fwd_  = this->get_parameter("left_fwd").as_int();
    left_rev_  = this->get_parameter("left_rev").as_int();
    right_fwd_ = this->get_parameter("right_fwd").as_int();
    right_rev_ = this->get_parameter("right_rev").as_int();

    gpiochip_   = this->get_parameter("gpiochip").as_string();
    pwm_hz_     = this->get_parameter("pwm_hz").as_int();
    timeout_ms_ = this->get_parameter("timeout_ms").as_int();

    inv_l_ = this->get_parameter("invert_left").as_bool();
    inv_r_ = this->get_parameter("invert_right").as_bool();

    chip_ = std::make_unique<gpiod::chip>(gpiochip_);

    req_output(left_fwd_);
    req_output(left_rev_);
    req_output(right_fwd_);
    req_output(right_rev_);

    stop_all();

    sub_ = this->create_subscription<std_msgs::msg::Int16MultiArray>(
      "/motor_cmd", 10,
      [this](std_msgs::msg::Int16MultiArray::SharedPtr msg) { on_motor_cmd(*msg); }
    );

    last_cmd_ = this->now();

    running_.store(true);
    pwm_thread_ = std::thread([this]{ pwm_loop(); });

    watchdog_ = this->create_wall_timer(50ms, [this]{
      auto dt_ms = (this->now() - last_cmd_).nanoseconds() / 1000000;
      if (dt_ms > timeout_ms_) {
        std::lock_guard<std::mutex> lk(cmd_mtx_);
        target_l_ = 0;
        target_r_ = 0;
      }
    });

    RCLCPP_INFO(
      get_logger(),
      "motor_driver READY | L:(%d,%d) R:(%d,%d) pwm=%dHz topic=/motor_cmd",
      left_fwd_, left_rev_, right_fwd_, right_rev_, pwm_hz_
    );
  }

  ~MotorDriverNode() override {
    running_.store(false);
    if (pwm_thread_.joinable()) pwm_thread_.join();
    stop_all();
  }

private:
  void req_output(int gpio) {
    auto line = chip_->get_line(gpio);
    line.request({ "motor_driver", gpiod::line_request::DIRECTION_OUTPUT, 0 }, 0);
    lines_[gpio] = std::move(line);
  }

  void write_gpio(int gpio, int v) {
    auto it = lines_.find(gpio);
    if (it != lines_.end()) it->second.set_value(v);
  }

  void stop_all() {
    write_gpio(left_fwd_, 0);
    write_gpio(left_rev_, 0);
    write_gpio(right_fwd_, 0);
    write_gpio(right_rev_, 0);
  }

  void on_motor_cmd(const std_msgs::msg::Int16MultiArray & msg) {
    if (msg.data.size() < 2) return;

    last_cmd_ = this->now();

    int l = clampi(msg.data[0], -1000, 1000);
    int r = clampi(msg.data[1], -1000, 1000);

    if (inv_l_) l = -l;
    if (inv_r_) r = -r;

    std::lock_guard<std::mutex> lk(cmd_mtx_);
    target_l_ = l;
    target_r_ = r;
  }

  struct Chan {
    int dir;   // -1,0,+1
    int duty;  // 0..1000
    int fwd;
    int rev;
  };

  static inline int abs_i(int x) { return x < 0 ? -x : x; }

  Chan compute_chan(int val, int fwd, int rev) {
    Chan c{};
    c.fwd = fwd;
    c.rev = rev;

    val = clampi(val, -1000, 1000);
    if (val == 0) {
      c.dir = 0;
      c.duty = 0;
      // iki pini de kapat
      write_gpio(fwd, 0);
      write_gpio(rev, 0);
      return c;
    }

    c.dir = (val > 0) ? 1 : -1;
    c.duty = abs_i(val); // 0..1000

    // karşı yön pinini kapat (shoot-through önlemi)
    if (c.dir > 0) write_gpio(rev, 0);
    else           write_gpio(fwd, 0);

    return c;
  }

  void set_on(const Chan& c) {
    if (c.duty <= 0 || c.dir == 0) return;
    if (c.dir > 0) write_gpio(c.fwd, 1);
    else           write_gpio(c.rev, 1);
  }

  void set_off(const Chan& c) {
    // hangi yönde olursa olsun iki pini de kapatmak güvenli
    write_gpio(c.fwd, 0);
    write_gpio(c.rev, 0);
  }

  void pwm_loop() {
    using clock = std::chrono::steady_clock;

    const int hz = std::max(1, pwm_hz_);
    const auto period = std::chrono::duration_cast<std::chrono::microseconds>(1s) / hz;

    while (running_.load()) {
      int l, r;
      {
        std::lock_guard<std::mutex> lk(cmd_mtx_);
        l = target_l_;
        r = target_r_;
      }

      const auto t0 = clock::now();

      // her kanal için dir + duty hesapla
      Chan L = compute_chan(l, left_fwd_, left_rev_);
      Chan R = compute_chan(r, right_fwd_, right_rev_);

      // duty -> on_time (mikrosaniye)
      const int64_t period_us = period.count();
      int64_t onL_us = (period_us * L.duty) / 1000;
      int64_t onR_us = (period_us * R.duty) / 1000;

      // ON: ikisini de başlat
      set_on(L);
      set_on(R);

      // farklı duty uygulaması: hangisi önce bitecekse onu önce kapat
      int64_t first_us = std::min(onL_us, onR_us);
      int64_t second_us = std::max(onL_us, onR_us);

      // first_us kadar bekle
      if (first_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(first_us));

      // first kapanış
      if (onL_us == first_us) set_off(L);
      if (onR_us == first_us) set_off(R);

      // ikinci kapanışa kadar kalan süre
      int64_t mid_us = second_us - first_us;
      if (mid_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(mid_us));

      // second kapanış (hala açıksa kapat)
      if (onL_us == second_us) set_off(L);
      if (onR_us == second_us) set_off(R);

      // periyodun kalanını bekle
      const auto spent = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0);
      if (spent < period) std::this_thread::sleep_for(period - spent);
    }
  }

private:
  int left_fwd_{13}, left_rev_{26};
  int right_fwd_{16}, right_rev_{25};

  std::string gpiochip_{"gpiochip4"};
  int pwm_hz_{200};
  int timeout_ms_{500};
  bool inv_l_{false}, inv_r_{false};

  rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr watchdog_;
  rclcpp::Time last_cmd_;

  std::unique_ptr<gpiod::chip> chip_;
  std::map<int, gpiod::line> lines_;

  std::atomic<bool> running_{false};
  std::thread pwm_thread_;
  std::mutex cmd_mtx_;
  int target_l_{0};
  int target_r_{0};
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MotorDriverNode>());
  rclcpp::shutdown();
  return 0;
}