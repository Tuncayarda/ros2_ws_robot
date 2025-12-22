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

    gpiochip_  = this->get_parameter("gpiochip").as_string();
    pwm_hz_    = this->get_parameter("pwm_hz").as_int();
    timeout_ms_= this->get_parameter("timeout_ms").as_int();

    inv_l_ = this->get_parameter("invert_left").as_bool();
    inv_r_ = this->get_parameter("invert_right").as_bool();

    // ===== GPIO OPEN =====
    chip_ = std::make_unique<gpiod::chip>(gpiochip_);

    req_output(left_fwd_);
    req_output(left_rev_);
    req_output(right_fwd_);
    req_output(right_rev_);

    stop_all();

    sub_ = this->create_subscription<std_msgs::msg::Int16MultiArray>(
      "/motor_cmd", 10,
      [this](std_msgs::msg::Int16MultiArray::SharedPtr msg) {
        on_motor_cmd(*msg);
      }
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
    line.request(
      { "motor_driver", gpiod::line_request::DIRECTION_OUTPUT, 0 }, 0
    );
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

  void pwm_loop() {
    using clock = std::chrono::steady_clock;

    const int hz = std::max(1, pwm_hz_);
    const auto period =
      std::chrono::duration_cast<std::chrono::microseconds>(1s) / hz;

    while (running_.load()) {
      int l, r;
      {
        std::lock_guard<std::mutex> lk(cmd_mtx_);
        l = target_l_;
        r = target_r_;
      }

      auto t0 = clock::now();

      auto apply = [&](int val, int fwd, int rev) {
        if (val == 0) {
          write_gpio(fwd, 0);
          write_gpio(rev, 0);
          return std::pair<int,int>(0, 0);
        }

        val = clampi(val, -1000, 1000);
        int dir = (val > 0) ? 1 : -1;
        int duty = std::abs(val); // 0..1000

        if (dir > 0) write_gpio(rev, 0);
        else         write_gpio(fwd, 0);

        return std::pair<int,int>(dir, duty);
      };

      auto [ld, lduty] = apply(l, left_fwd_, left_rev_);
      auto [rd, rduty] = apply(r, right_fwd_, right_rev_);

      int on_duty = std::max(lduty, rduty);
      double on_frac = on_duty / 1000.0;
      auto on_time =
        std::chrono::duration_cast<std::chrono::microseconds>(period * on_frac);

      // ON
      if (ld > 0) write_gpio(left_fwd_, 1);
      if (ld < 0) write_gpio(left_rev_, 1);
      if (rd > 0) write_gpio(right_fwd_, 1);
      if (rd < 0) write_gpio(right_rev_, 1);

      if (on_time.count() > 0) std::this_thread::sleep_for(on_time);

      // OFF
      if (ld > 0) write_gpio(left_fwd_, 0);
      if (ld < 0) write_gpio(left_rev_, 0);
      if (rd > 0) write_gpio(right_fwd_, 0);
      if (rd < 0) write_gpio(right_rev_, 0);

      auto spent =
        std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0);
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