#include <cstddef>
#include <vector>

class ProgressTracker {
public:
    ProgressTracker(size_t total, size_t nthreads)
        : total_(total), nthreads_(nthreads), progress_(nthreads, 0)
    {
        raw_ptr_ = progress_.data();
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

    size_t total() const { return total_; }
    size_t nthreads() const { return nthreads_; }

private:
    size_t total_;
    size_t nthreads_;
    std::vector<size_t> progress_;
    size_t* raw_ptr_;  // 给算子用的裸指针（非原子）
};