#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>

#include <gpiod.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

class ServoSoftPwmNode : public rclcpp::Node
{
public:
  ServoSoftPwmNode() : Node("servo_driver_node")
  {
    // ---------- Params ----------
    chip_path_   = declare_parameter<std::string>("chip", "/dev/gpiochip4");
    line_offset_ = declare_parameter<int>("gpio", 12);          // BCM numarası = gpiochip line offset (Pi'de genelde aynı)
    topic_       = declare_parameter<std::string>("topic", "/servo/angle_deg");

    min_us_    = declare_parameter<int>("min_us", 500);
    max_us_    = declare_parameter<int>("max_us", 2500);
    min_deg_   = declare_parameter<double>("min_deg", 0.0);
    max_deg_   = declare_parameter<double>("max_deg", 180.0);

    period_us_     = declare_parameter<int>("period_us", 20000); // 50Hz
    max_step_us_   = declare_parameter<int>("max_step_us", 15);  // yumuşatma
    rt_priority_   = declare_parameter<int>("rt_priority", 60);   // root/CAP_SYS_NICE varsa işe yarar

    // ---------- GPIO (libgpiod) init ----------
    chip_ = gpiod_chip_open(chip_path_.c_str());
    if (!chip_) {
      throw std::runtime_error("gpiod_chip_open failed: " + chip_path_);
    }

    line_ = gpiod_chip_get_line(chip_, line_offset_);
    if (!line_) {
      gpiod_chip_close(chip_);
      throw std::runtime_error("gpiod_chip_get_line failed (gpio offset yanlış olabilir)");
    }

    int rc = gpiod_line_request_output(line_, "servo_softpwm_node", 0);
    if (rc != 0) {
      gpiod_chip_close(chip_);
      throw std::runtime_error("gpiod_line_request_output failed (permission?)");
    }

    // başlangıç: orta
    target_us_.store((min_us_ + max_us_) / 2);
    current_us_.store(target_us_.load());

    // ---------- Subscriber ----------
    sub_ = create_subscription<std_msgs::msg::Float32>(
      topic_, 10,
      [this](const std_msgs::msg::Float32 &msg) {
        const double deg = clamp(msg.data, min_deg_, max_deg_);
        const double t = (deg - min_deg_) / std::max(1e-9, (max_deg_ - min_deg_));
        const int pulse = (int)std::lround(min_us_ + t * (max_us_ - min_us_));
        target_us_.store(std::clamp(pulse, min_us_, max_us_));
      }
    );

    // ---------- Worker ----------
    running_.store(true);
    worker_ = std::thread([this] { this->worker_loop(); });

    RCLCPP_INFO(get_logger(),
      "servo_softpwm_node ready. chip=%s gpio=%d topic=%s pulse=[%dus..%dus] period=%dus max_step_us=%d",
      chip_path_.c_str(), line_offset_, topic_.c_str(), min_us_, max_us_, period_us_, max_step_us_);
  }

  ~ServoSoftPwmNode() override
  {
    running_.store(false);
    if (worker_.joinable()) worker_.join();

    // GPIO low + release
    if (line_) {
      gpiod_line_set_value(line_, 0);
      gpiod_line_release(line_);
      line_ = nullptr;
    }
    if (chip_) {
      gpiod_chip_close(chip_);
      chip_ = nullptr;
    }
  }

private:
  static double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
  }

  void try_set_realtime_priority()
  {
    setpriority(PRIO_PROCESS, 0, -10);

    sched_param sch{};
    sch.sched_priority = std::clamp(rt_priority_, 1, 99);
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch);
    // root/CAP_SYS_NICE yoksa başarısız olur; problem değil.
  }

  static void sleep_ns(long ns)
  {
    timespec ts{};
    ts.tv_sec  = ns / 1000000000L;
    ts.tv_nsec = ns % 1000000000L;
    clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);
  }

  void worker_loop()
  {
    try_set_realtime_priority();

    const long period_ns = (long)period_us_ * 1000L;

    while (running_.load()) {
      // yumuşatma (her 50Hz frame'de yaklaş)
      int cur = current_us_.load();
      int tgt = target_us_.load();
      if (cur != tgt) {
        int diff = tgt - cur;
        int step = std::clamp(diff, -max_step_us_, max_step_us_);
        cur = std::clamp(cur + step, min_us_, max_us_);
        current_us_.store(cur);
      }

      const long high_ns = (long)current_us_.load() * 1000L;
      const long low_ns  = std::max(0L, period_ns - high_ns);

      // HIGH
      gpiod_line_set_value(line_, 1);
      sleep_ns(high_ns);

      // LOW
      gpiod_line_set_value(line_, 0);
      sleep_ns(low_ns);
    }
  }

private:
    // params
    std::string chip_path_{"/dev/gpiochip4"};
    int line_offset_{12};
    std::string topic_{"/servo/angle_deg"};

    int min_us_{500}, max_us_{2500};
    double min_deg_{0.0}, max_deg_{180.0};

    int period_us_{20000};
    int max_step_us_{15};
    int rt_priority_{60};

    // gpio
    gpiod_chip *chip_{nullptr};
    gpiod_line *line_{nullptr};

    // state
    std::atomic<bool> running_{false};
    std::atomic<int> target_us_{1500};
    std::atomic<int> current_us_{1500};

    std::thread worker_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ServoSoftPwmNode>());
  rclcpp::shutdown();
  return 0;
}