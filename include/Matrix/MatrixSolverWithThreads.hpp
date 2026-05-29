#pragma once
#include "../threadPool/ThreadPool.hpp"
#include "./Matrix.hpp"
#include <vector>

template <typename T>
class MatrixSolverWithThreads{
    private:
        const static int NUM_THREADS = 4;
        ThreadPool _pool;

        void _taskSum( Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const std::vector<int> range){
            int rows = range[1] - range[0] + 1;
            for (int i = 0; i < rows; i++)
            {
                int start = range[0] + i;
                for (size_t j = 0; j < result.cols; j++)
                {
                    result[start][j] = A[start][j] + B[start][j];
                }
            }
        }

        void _taskSumWithVector( Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const std::vector<int> range){
            int rows = range[1] - range[0] + 1;
            for (int i = 0; i < rows; i++)
            {
                int start = range[0] + i;
                for (size_t j = 0; j < result.cols; j++)
                {
                    result[start][j] = A[start][j] + B[start][0];
                }
            }
        }

        void _taskSub( Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const std::vector<int> range){
            int rows = range[1] - range[0] + 1;
            for (int i = 0; i < rows; i++)
            {
                int start = range[0] + i;
                for (size_t j = 0; j < result.cols; j++)
                {
                    result[start][j] = A[start][j] - B[start][j];
                }
            }
        }

        void _taskSubWithVector( Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const std::vector<int> range){
            int rows = range[1] - range[0] + 1;
            for (int i = 0; i < rows; i++)
            {
                int start = range[0] + i;
                for (size_t j = 0; j < result.cols; j++)
                {
                    result[start][j] = A[start][j] - B[start][0];
                }
            }
        }

        void _taskMul( Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const std::vector<int> range){
            int rows = range[1] - range[0] + 1;
            for (int i = 0; i < rows; i++)
            {
                int start = range[0] + i;
                for (size_t j = 0; j < result.cols; j++)
                {
                    for (size_t k = 0; k < A.cols; k++)
                    {
                        result[start][j] += A[start][k] * B[k][j];
                    } 
                }
            }
        }


        Matrix<int> _calculateTileSize(const Matrix<T> &A){
            int numRows = 0;
            int aux = 0;
            if(A.rows > 100 * MatrixSolverWithThreads::NUM_THREADS){
                int numTask = (int)(A.rows / 100);
                numRows = 100;
                Matrix<int> ranges(numTask,2);
                for (int i = 0; i < numTask; i++)
                {
                    aux = i*numRows;
                    ranges[i][0] = aux;
                    if(i != numTask - 1){
                        ranges[i][1] = (aux - 1) + numRows;
                        continue;
                    }
                    ranges[i][1] = (aux - 1) + numRows + (A.rows % 100);
                }

                return ranges;
            }

            Matrix<int> ranges(MatrixSolverWithThreads::NUM_THREADS,2);
            numRows = (int)(A.rows / MatrixSolverWithThreads::NUM_THREADS);
            for (int i = 0; i < MatrixSolverWithThreads::NUM_THREADS; i++)
            {
                aux = i*numRows;
                ranges[i][0] = aux;
                if(i != MatrixSolverWithThreads::NUM_THREADS - 1){
                    ranges[i][1] = (aux - 1) + numRows;
                    continue;
                }
                ranges[i][1] = (aux - 1) + numRows + (A.rows % MatrixSolverWithThreads::NUM_THREADS);
            }
            return ranges;
        }

        void _taskGradient( Matrix<T> &result, const Matrix<T> &A, const std::vector<int> range){
            int rows = range[1] - range[0] + 1;
            for (int i = 0; i < rows; i++)
            {
                int start = range[0] + i;
                for (size_t j = 0; j < A.cols; j++)
                {
                    result[start][0] += A[start][j]; 
                }
            }
        }

    public:
        MatrixSolverWithThreads():_pool(MatrixSolverWithThreads::NUM_THREADS){};

        Matrix<T> sum(const Matrix<T> &A, const Matrix<T> &B){
            if(A.rows * A.cols > 20000) return A + B;

            if (A.rows != B.rows || A.cols != B.cols){
                throw std::runtime_error("Matrix dimensions must match for addition.");
            }
            Matrix<int> ranges = this->_calculateTileSize(A);
            Matrix<T> result(A.rows,A.cols);

            for (size_t i = 0; i < ranges.rows; i++)
            {
                this->_pool.addTask([&,i]{
                    _taskSum(result,A,B,ranges.getRowAsVector(i)); 
                    });   
            }
            this->_pool.wait();
            return result;
        }

        Matrix<T> sumWithVector(const Matrix<T> &A, const Matrix<T> &B){
            if (A.rows != B.rows || B.cols != 1){
                throw std::runtime_error("Matrix dimensions must match for addition.");
            }

            Matrix<int> ranges = this->_calculateTileSize(A);
            Matrix<T> result(A.rows,A.cols);
            for (size_t i = 0; i < ranges.rows; i++)
            {
                this->_pool.addTask([&,i]{
                    _taskSumWithVector(result,A,B,ranges.getRowAsVector(i)); 
                    });   
            }
            this->_pool.wait();
            return result;
        }


        Matrix<T> sub(const Matrix<T> &A, const Matrix<T> &B){
            if(A.rows * A.cols > 20000) return A - B;

            if (A.rows != B.rows || A.cols != B.cols){
                throw std::runtime_error("Matrix dimensions must match for subtraction.");
            }
            Matrix<int> ranges = this->_calculateTileSize(A);
            Matrix<T> result(A.rows,A.cols);

            for (size_t i = 0; i < ranges.rows; i++)
            {
                this->_pool.addTask([&,i]{
                    _taskSub(result,A,B,ranges.getRowAsVector(i)); 
                    });   
            }
            this->_pool.wait();
            return result;
        }

        Matrix<T> subWithVector(const Matrix<T> &A, const Matrix<T> &B){

            if (A.rows != B.rows || B.cols != 1){
                throw std::runtime_error("Matrix dimensions must match for subtraction.");
            }
            Matrix<int> ranges = this->_calculateTileSize(A);
            Matrix<T> result(A.rows,A.cols);

            for (size_t i = 0; i < ranges.rows; i++)
            {
                this->_pool.addTask([&,i]{
                    _taskSubWithVector(result,A,B,ranges.getRowAsVector(i)); 
                    });   
            }
            this->_pool.wait();
            return result;
        }

        Matrix<T> mul(const Matrix<T> &A, const Matrix<T> &B){
            if (A.cols != B.rows)
            {
                throw std::runtime_error("Matrix dimensions must match rows = cols");
            }

            Matrix<int> ranges = this->_calculateTileSize(A);
            Matrix<T> result(A.rows,B.cols);

            for (size_t i = 0; i < ranges.rows; i++)
            {
                this->_pool.addTask([&,i]{
                    _taskMul(result,A,B,ranges.getRowAsVector(i)); 
                    });   
            }
            this->_pool.wait();
            return result;
        }

        Matrix<T> gradient(const Matrix<T> &A){

            Matrix<int> ranges = this->_calculateTileSize(A);
            Matrix<T> result(A.rows,1);

            for (size_t i = 0; i < ranges.rows; i++)
            {
                this->_pool.addTask([&,i]{
                    _taskGradient(result,A,ranges.getRowAsVector(i)); 
                    });   
            }
            this->_pool.wait();
            for (size_t i = 0; i < A.rows; i++)
            {
                result[i][0] = result[i][0] / A.cols; 
            }
            
            return result;
        }
};