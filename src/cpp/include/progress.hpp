#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <indicators/progress_bar.hpp>
#include <indicators/block_progress_bar.hpp>

class ProgressTracker {
public:
    ProgressTracker(size_t total, size_t nthreads, bool show_progress = false)
        : total_(total), nthreads_(nthreads), progress_(nthreads, 0), show_progress_(show_progress)
    {
        raw_ptr_ = progress_.data();
        
        if (show_progress_) {
            // 创建进度条
            progress_bar_ = std::make_unique<indicators::BlockProgressBar>(
                indicators::option::BarWidth{50},
                indicators::option::Start{"["},
                indicators::option::End{"]"},
                indicators::option::PrefixText{"hpdex mwu: "},
                indicators::option::PostfixText{""},
                indicators::option::ShowPercentage{true},
                indicators::option::ShowElapsedTime{true},
                indicators::option::ShowRemainingTime{true},
                indicators::option::Completed{false},
                indicators::option::SavedStartTime{false},
                indicators::option::MaxPostfixTextLen{0},
                indicators::option::FontStyles{std::vector<indicators::FontStyle>{}},
                indicators::option::MaxProgress{total_},
                indicators::option::Stream{std::cout},
                indicators::option::ForegroundColor{indicators::Color::white}
            );
        }
    }
    
    // 析构函数，确保线程正确退出
    ~ProgressTracker() {
        stop_progress_display();
    }

    size_t* ptr() {
        return raw_ptr_;
    }

    template <typename Callback>
    size_t aggregate_and_notify(Callback&& callback) const {
        size_t sum = 0;
        for (size_t i = 0; i < nthreads_; ++i) {
            sum += progress_[i];
        }
        callback(sum);
        return sum;
    }
    
    // 启动进度条显示线程
    void start_progress_display() {
        if (show_progress_ && progress_bar_) {
            progress_thread_ = std::make_unique<std::thread>([this]() {
                while (!stop_progress_) {
                    size_t sum = 0;
                    for (size_t i = 0; i < nthreads_; ++i) {
                        sum += progress_[i];
                    }
                    
                    {
                        std::lock_guard<std::mutex> lock(progress_mutex_);
                        progress_bar_->set_progress(sum);
                    }
                    
                    // 如果已完成，退出线程
                    if (sum >= total_) {
                        break;
                    }
                    
                    // 每100ms更新一次
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            });
        }
    }
    
    // 停止进度条显示
    void stop_progress_display() {
        if (progress_thread_) {
            stop_progress_ = true;
            progress_thread_->join();
            progress_thread_.reset();
        }
    }
    
    // 完成进度条
    void complete() {
        if (show_progress_ && progress_bar_) {
            // 设置最终进度
            {
                std::lock_guard<std::mutex> lock(progress_mutex_);
                progress_bar_->set_progress(total_);
            }
            // 停止显示线程
            stop_progress_display();
        }
    }

    size_t total() const { return total_; }
    size_t nthreads() const { return nthreads_; }

private:
    size_t total_;
    size_t nthreads_;
    std::vector<size_t> progress_;
    size_t* raw_ptr_;  // 给算子用的裸指针（非原子）
    bool show_progress_;
    mutable std::unique_ptr<indicators::BlockProgressBar> progress_bar_;
    mutable std::mutex progress_mutex_;  // 保护进度条更新的互斥锁
    std::unique_ptr<std::thread> progress_thread_;  // 进度条显示线程
    std::atomic<bool> stop_progress_{false};  // 停止进度条显示的标志
};