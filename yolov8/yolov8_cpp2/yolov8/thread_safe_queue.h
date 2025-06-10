// thread_safe_queue.h
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t maxSize = 0) : max_size(maxSize) {}

    // Add an element to the queue. Blocks if the queue is full and maxSize > 0.
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size > 0) {
            cond_not_full_.wait(lock, [this] { return queue_.size() < max_size; });
        }
        queue_.push(std::move(item));
        lock.unlock();
        cond_not_empty_.notify_one();
    }

    // Try to add an element. Returns false if the queue is full.
    bool try_push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size > 0 && queue_.size() >= max_size) {
            return false;
        }
        queue_.push(std::move(item));
        lock.unlock();
        cond_not_empty_.notify_one();
        return true;
    }

    // Wait and pop an element from the queue.
    T wait_and_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_not_empty_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        cond_not_full_.notify_one();
        return item;
    }

    // Try to pop an element. Returns an empty optional if the queue is empty.
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        cond_not_full_.notify_one();
        return item;
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_not_empty_;
    std::condition_variable cond_not_full_;
    size_t max_size;
};