#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <string>
#include <indicators/progress_bar.hpp>
#include <indicators/block_progress_bar.hpp>

class ProgressTracker {
public:
    ProgressTracker(size_t total, size_t nthreads, bool show_progress = false, const std::string& prefix = "hpdex mwu")
        : total_(total), nthreads_(nthreads), progress_(nthreads, 0), show_progress_(show_progress), 
          is_running_(false), should_stop_(false)
    {
        raw_ptr_ = progress_.data();
        
        if (show_progress_) {
            // 创建进度条
            progress_bar_ = std::make_unique<indicators::BlockProgressBar>(
                indicators::option::BarWidth{50},
                indicators::option::Start{"["},
                indicators::option::End{"]"},
                indicators::option::PrefixText{prefix + ": "},
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
        if (show_progress_ && progress_bar_ && !is_running_) {
            should_stop_ = false;
            is_running_ = true;
            
            progress_thread_ = std::make_unique<std::thread>([this]() {
                size_t last_progress = 0;
                auto last_update = std::chrono::steady_clock::now();
                
                while (!should_stop_) {
                    // 统计所有线程的进度
                    size_t current_progress = 0;
                    for (size_t i = 0; i < nthreads_; ++i) {
                        current_progress += progress_[i];
                    }
                    
                    // 只有当进度发生变化时才更新进度条
                    if (current_progress != last_progress) {
                        {
                            std::lock_guard<std::mutex> lock(progress_mutex_);
                            progress_bar_->set_progress(std::min(current_progress, total_));
                        }
                        last_progress = current_progress;
                        last_update = std::chrono::steady_clock::now();
                    }
                    
                    // 如果已完成，退出线程
                    if (current_progress >= total_) {
                        break;
                    }
                    
                    // 检查是否长时间没有更新（超时检测）
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
                    if (elapsed.count() > 30) {  // 30秒超时
                        std::cout << "\n警告: 进度条超时，可能存在死锁\n";
                        break;
                    }
                    
                    // 每200ms检查一次
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                
                // 确保进度条显示最终状态
                {
                    std::lock_guard<std::mutex> lock(progress_mutex_);
                    progress_bar_->set_progress(total_);
                }
                
                is_running_ = false;
            });
        }
    }
    
    // 停止进度条显示
    void stop_progress_display() {
        if (progress_thread_ && is_running_) {
            should_stop_ = true;
            if (progress_thread_->joinable()) {
                progress_thread_->join();
            }
            progress_thread_.reset();
            is_running_ = false;
        }
    }
    
    // 完成进度条
    void complete() {
        if (show_progress_ && progress_bar_) {
            // 等待进度条线程自然结束
            if (is_running_) {
                should_stop_ = true;
                if (progress_thread_ && progress_thread_->joinable()) {
                    progress_thread_->join();
                }
                progress_thread_.reset();
                is_running_ = false;
            }
            
            // 设置最终进度
            {
                std::lock_guard<std::mutex> lock(progress_mutex_);
                progress_bar_->set_progress(total_);
            }
            
            // 强制刷新输出，确保进度条正确显示
            std::cout.flush();
        }
    }
    
    // 获取当前总进度
    size_t get_current_progress() const {
        size_t sum = 0;
        for (size_t i = 0; i < nthreads_; ++i) {
            sum += progress_[i];
        }
        return sum;
    }
    
    // 检查是否正在运行
    bool is_running() const {
        return is_running_;
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
    std::atomic<bool> should_stop_{false};  // 停止进度条显示的标志
    std::atomic<bool> is_running_{false};   // 进度条是否正在运行
};