#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <indicators/block_progress_bar.hpp>

class ProgressTracker {
public:
    ProgressTracker(size_t total, size_t nthreads)
        : nthreads_(nthreads), progress_(nthreads, 0) {
        raw_ptr_ = progress_.data();
    }

    size_t* ptr() { return raw_ptr_; }

    size_t aggregate() const {
        size_t sum = 0;
        for (size_t i = 0; i < nthreads_; ++i) {
            sum += progress_[i];
        }
        return sum;
    }

    size_t nthreads() const { return nthreads_; }

private:
    size_t nthreads_;
    std::vector<size_t> progress_;
    size_t* raw_ptr_;  // 非原子，不加锁，由外部控制
};

class ProgressBar {
public:
    ProgressBar(size_t total, const ProgressTracker& tracker, int interval_ms = 200,
                const std::string& prefix = "progress")
        : total_(total), tracker_(tracker), interval_(interval_ms), running_(false)
    {
        progress_bar_ = std::make_unique<indicators::BlockProgressBar>(
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::End{"]"},
            indicators::option::PrefixText{prefix + ": "},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::MaxProgress{total_},
            indicators::option::Stream{std::cout},
            indicators::option::ForegroundColor{indicators::Color::white}
        );
    }

    void start() {
        if (running_) return;
        running_ = true;
        worker_ = std::thread([this]() {
            while (running_) {
                size_t current = tracker_.aggregate();
                {
                    progress_bar_->set_progress(std::min(current, total_));
                }
                if (current >= total_) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_));
            }
            // 确保完成
            progress_bar_->set_progress(total_);
        });
    }

    void stop() {
        running_ = false;
        if (worker_.joinable()) worker_.join();
    }

    ~ProgressBar() {
        stop();
    }

private:
    size_t total_;
    const ProgressTracker& tracker_;
    int interval_;
    std::atomic<bool> running_;
    std::thread worker_;
    std::unique_ptr<indicators::BlockProgressBar> progress_bar_;
};
