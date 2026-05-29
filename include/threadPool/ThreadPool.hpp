#pragma once

#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <vector>


class ThreadPool{
    private:
        std::mutex _mtx;
        std::condition_variable _cv;
       
        std::queue<std::function<void()>> _tasks;
        std::vector<std::thread> _workers;

        std::condition_variable _taskFinished;
        int _activeTasks = 0;
        int _queueTasks = 0;

        bool _flagAlive = true;
        
        void _createThread(){
            this->_workers.emplace_back(
                std::thread([&]{
                    while (true)
                    {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(_mtx);
                            
                            while (_tasks.empty() && _flagAlive) _cv.wait(lock);
                            
                            if(_tasks.empty() && !_flagAlive) return;
                            task = std::move(_tasks.front());
                            _tasks.pop();
                            _queueTasks--;
                            _activeTasks++;
                        }
                        task();
                        {
                            std::unique_lock<std::mutex> lock(_mtx);
                            _activeTasks--;
                            if(_activeTasks == 0 && _queueTasks == 0) _taskFinished.notify_all();
                        }
                    }
                })
            );
        }


    public:
        
        ThreadPool(unsigned int threadsCount){
            if( threadsCount == 0) threadsCount = 2;

            for (size_t i = 0; i < threadsCount; i++)
            {
                this->_createThread();
            }
            
        }
        ThreadPool(){
            unsigned int threadsCount = std::thread::hardware_concurrency();
            if(  threadsCount == 0) threadsCount = 2;
            
            for (size_t i = 0; i < threadsCount; i++)
            {
                this->_createThread();
            }
            
        }

        ~ThreadPool() {
            closeThreadPool();
            for (auto &t : this->_workers) {
                if (t.joinable()) t.join();
            }
        }

        void addTask(std::function<void()> task) {
            {
                std::unique_lock<std::mutex> lock(this->_mtx);
                this->_tasks.push(std::move(task));
                this->_queueTasks ++;
            }
            this->_cv.notify_one();
        }

        

        void closeThreadPool(){
            {
                std::unique_lock<std::mutex> lock(this->_mtx);
                this->_flagAlive = false;
            }
            this->_cv.notify_all();
        }

        void wait(){
            {
                std::unique_lock<std::mutex> lock(this->_mtx);
                while (this->_activeTasks != 0 || this->_queueTasks != 0) _taskFinished.wait(lock);     
            }
        }
};