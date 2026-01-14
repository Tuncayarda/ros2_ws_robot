#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>

#include <gpiod.h>

#include <algorithm>
#include <atomic>
#include <cmath>
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
    line_offset_ = declare_parameter<int>("gpio", 12);
    topic_       = declare_parameter<std::string>("topic", "/servo/angle_deg");

    min_us_    = declare_parameter<int>("min_us", 500);
    max_us_    = declare_parameter<int>("max_us", 2500);

    min_deg_   = declare_parameter<double>("min_deg", 0.0);
    max_deg_   = declare_parameter<double>("max_deg", 180.0);

    center_deg_ = declare_parameter<double>("center_deg", 100.0);

    period_us_   = declare_parameter<int>("period_us", 20000); // 50Hz
    max_step_us_ = declare_parameter<int>("max_step_us", 15);
    rt_priority_ = declare_parameter<int>("rt_priority", 60);

    // PWM kesme mantığı
    hold_ms_      = declare_parameter<int>("hold_ms", 250);     // hedefe geldikten sonra kaç ms daha basıp bırakacak
    deadband_us_  = declare_parameter<int>("deadband_us", 6);   // hedefe "yakın" kabul aralığı (us)

    // ---------- GPIO ----------
    chip_ = gpiod_chip_open(chip_path_.c_str());
    if (!chip_) throw std::runtime_error("gpiod_chip_open failed");

    line_ = gpiod_chip_get_line(chip_, line_offset_);
    if (!line_) throw std::runtime_error("gpiod_chip_get_line failed");

    if (gpiod_line_request_output(line_, "servo_softpwm_node", 0) != 0)
      throw std::runtime_error("gpiod_line_request_output failed");

    target_us_.store((min_us_ + max_us_) / 2);
    current_us_.store(target_us_.load());

    active_.store(false);
    last_cmd_ns_.store(0);

    // ---------- Subscriber ----------
    sub_ = create_subscription<std_msgs::msg::Float32>(
      topic_, 10,
      [this](const std_msgs::msg::Float32 &msg)
      {
        // MOBILDEN GELEN: -40..0..+40
        const double offset = -msg.data;

        double desired_deg = center_deg_ + offset;
        desired_deg = std::clamp(desired_deg, min_deg_, max_deg_);

        const double t =
          (desired_deg - min_deg_) /
          std::max(1e-6, (max_deg_ - min_deg_));

        const int pulse =
          (int)std::lround(min_us_ + t * (max_us_ - min_us_));

        target_us_.store(std::clamp(pulse, min_us_, max_us_));

        // PWM’i aç ve hold timer’ı başlat
        last_cmd_ns_.store(now_ns());
        active_.store(true);
      }
    );

    running_.store(true);
    worker_ = std::thread([this]{ worker_loop(); });

    RCLCPP_INFO(get_logger(),
      "servo_driver ready | topic=%s center=%.1fdeg range=[%.1f..%.1f] hold_ms=%d deadband_us=%d",
      topic_.c_str(), center_deg_, min_deg_, max_deg_, hold_ms_, deadband_us_);
  }

  ~ServoSoftPwmNode()
  {
    running_.store(false);
    if (worker_.joinable()) worker_.join();

    gpiod_line_set_value(line_, 0);
    gpiod_line_release(line_);
    gpiod_chip_close(chip_);
  }

private:
  static long long now_ns()
  {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
  }

  static void sleep_ns(long ns)
  {
    if (ns <= 0) return;
    timespec ts{ ns / 1'000'000'000L, ns % 1'000'000'000L };
    clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);
  }

  void try_rt()
  {
    setpriority(PRIO_PROCESS, 0, -10);
    sched_param sch{};
    sch.sched_priority = std::clamp(rt_priority_, 1, 99);
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch);
  }

  void pwm_frame(int pulse_us)
  {
    long period_ns = period_us_ * 1000L;
    long high_ns   = pulse_us * 1000L;

    gpiod_line_set_value(line_, 1);
    sleep_ns(high_ns);
    gpiod_line_set_value(line_, 0);
    sleep_ns(period_ns - high_ns);
  }

  void worker_loop()
  {
    try_rt();

    const long long hold_ns = (long long)hold_ms_ * 1'000'000LL;

    while (running_.load())
    {
      if (!active_.load())
      {
        // PWM kapalı: çizgiyi LOW bırak
        gpiod_line_set_value(line_, 0);
        sleep_ns(5'000'000); // 5ms
        continue;
      }

      int cur = current_us_.load();
      int tgt = target_us_.load();

      if (cur != tgt)
      {
        int diff = tgt - cur;
        int step = std::clamp(diff, -max_step_us_, max_step_us_);
        current_us_.store(cur + step);
        cur = current_us_.load();
      }

      // Frame bas (servo hareket etsin)
      pwm_frame(cur);

      // Hedefe yaklaşıldıysa ve hold süresi geçtiyse PWM kes
      const int err_us = std::abs(cur - tgt);
      if (err_us <= deadband_us_)
      {
        long long t0 = last_cmd_ns_.load();
        if (t0 != 0 && (now_ns() - t0) >= hold_ns)
        {
          active_.store(false);
          // bir sonraki döngüde LOW'a geçecek
        }
      }
    }
  }

private:
  // params
  std::string chip_path_;
  int line_offset_;
  std::string topic_;

  int min_us_, max_us_;
  double min_deg_, max_deg_;
  double center_deg_;

  int period_us_, max_step_us_, rt_priority_;
  int hold_ms_;
  int deadband_us_;

  // gpio
  gpiod_chip *chip_{nullptr};
  gpiod_line *line_{nullptr};

  // state
  std::atomic<bool> running_{false};
  std::atomic<bool> active_{false};
  std::atomic<int> target_us_{1500};
  std::atomic<int> current_us_{1500};

  // son komut zamanı (hold hesabı)
  std::atomic<long long> last_cmd_ns_{0};

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