#pragma once
#include "Layer.hpp"
#include <string>
#include "../Matrix/MatrixSolverWithThreads.hpp"

template <typename T = float>
class NeuralNetwork
{
    private:
    MatrixSolverWithThreads<T> solver;
    int batcheSize;
    int var_print_loss = 0;
    float loss = 10000000;
public:
    std::vector<int> shape;
    std::vector<Layer<T>> layers;

    T (*activationFunction)(T);
    T (*derivativeActivationFunction)(T);

    NeuralNetwork(std::vector<int> &shape, std::string a_func,int batcheSize):batcheSize(batcheSize)
    {
        this->activationFunction = NeuralNetwork<T>::sigmoid;
        if (a_func == "sigmoid")
        {
            this->activationFunction = NeuralNetwork<T>::sigmoid;
            this->derivativeActivationFunction = NeuralNetwork<T>::dSigmoid;
        }
        if (a_func == "RELU")
        {
            this->activationFunction = NeuralNetwork<T>::relu;
            this->derivativeActivationFunction = NeuralNetwork<T>::dRelu;
        }
        this->shape = shape;

        for (size_t i = 0; i < this->shape.size() - 1; i++)
        {
            this->layers.push_back(Layer<T>(this->shape[i], this->shape[i + 1],this->batcheSize));
        }
    }

    Matrix<T> forwardPass(const Matrix<T> &inputs)
    {
        Matrix<T> result;
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            if (i == 0)
            {
                result = this->layers[i].calculateLayer(inputs, this->activationFunction,this->solver);
                continue;
            }
            result = this->layers[i].calculateLayer(result, this->activationFunction,this->solver);
        }
        return result;
    }

    void backPropagation(const Matrix<T> &real)
    {
        // dC/dW = dC/dA * dA/dz * dZ/dW  -----  dZ/dW = a^L-1
        // dC/dB = dC/dA * dA/dz * dZ/dB  -----  dZ/dB = 1

        // delta^L = dC/dA^L * dA^L/dz^L
        // dC/dW = delta^L * dZ/dW  -----  dZ/dW = a^L-1
        // dC/dB = delta^L * dZ/dB  -----  dZ/dB = 1

        // for (int i = this->layers.size() - 1; i >= 0; i--)
        // {
        //     // a^L
        //     Matrix<T> aValuesLayer = this->layers[i].activationValues;
        //     if (this->layers.size() - 1 == i)
        //     {
        //         // dC/dA
        //         Matrix<T> dC = NeuralNetwork<T>::dErrCuaMed(aValuesLayer, real);

        //         // dA/dz
        //         Matrix<T> dA = aValuesLayer;
        //         dA.applyFuntion(this->derivativeActivationFunction);
        //         dC.hadamardProduct(dA);

        //         // Delta
        //         this->layers[i].delta = dC;
        //         continue;
        //     }

        //     // delta^L = (delta^L+1 * W^L+1) *  dA^L/dz^L
        //     Matrix<T> deltaLayerL = this->layers[i + 1].weights.t() * this->layers[i + 1].delta;
        //     // dA^L/dz^L
        //     Matrix<T> dA = aValuesLayer;
        //     dA.applyFuntion(this->derivativeActivationFunction);

        //     deltaLayerL.hadamardProduct(dA);
        //     this->layers[i].delta = deltaLayerL;
        // }


         for (int i = this->layers.size() - 1; i >= 0; i--)
        {
            // a^L
            Matrix<T> aValuesLayer = this->layers[i].activationValues;
            if (this->layers.size() - 1 == i)
            {
                // dC/dA
                Matrix<T> dC = NeuralNetwork<T>::dErrCuaMed(aValuesLayer, real);

                // dA/dz
                Matrix<T> dA = aValuesLayer;
                dA.applyFuntion(this->derivativeActivationFunction);
                dC.hadamardProduct(dA);

                // Delta
                this->layers[i].delta = dC;
                continue;
            }

            // delta^L = (delta^L+1 * W^L+1) *  dA^L/dz^L
            Matrix<T> deltaLayerL = solver.mul(this->layers[i + 1].weights.t(), this->layers[i + 1].delta);
            // dA^L/dz^L
            Matrix<T> dA = aValuesLayer;
            dA.applyFuntion(this->derivativeActivationFunction);

            deltaLayerL.hadamardProduct(dA);
            this->layers[i].delta = deltaLayerL;
        }
    }

    void gradientDescent(float learningRate, const Matrix<T> &inputs)
    {
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            // this->layers[i].bias = this->layers[i].bias - (this->layers[i].delta * learningRate);
            // if (i == 0)
            // {
            //     this->layers[i].weights = this->layers[i].weights - ((this->layers[i].delta * inputs) * learningRate);
            // }
            // else
            // {
            //     this->layers[i].weights = this->layers[i].weights - ((this->layers[i].delta * this->layers[i - 1].activationValues.t()) * learningRate);
            // }

            Matrix<T> db = solver.gradient(this->layers[i].delta);
            this->layers[i].bias = solver.sub(
                this->layers[i].bias, 
                db * (learningRate/this->batcheSize)
            );

            if (i == 0)
            {
                this->layers[i].weights = solver.sub(
                    this->layers[i].weights , 
                    solver.mul(this->layers[i].delta, inputs.t()) * (learningRate/this->batcheSize)
                );
            }
            else
            {
                this->layers[i].weights = solver.sub(
                    this->layers[i].weights, 
                    solver.mul(this->layers[i].delta, this->layers[i - 1].activationValues.t()) * (learningRate/this->batcheSize)
                );
            }
        }
    }

    void train(float learningRate, const Matrix<T> &inputs, const Matrix<T> &realValues)
    {
        
        for (size_t j = 0; j < inputs.rows; j++)
        {
            Matrix<T> rowInput = inputs.getRow(j);
            Matrix<T> rowRealValues = realValues.getRow(j);

            Matrix<T> result = this->forwardPass(rowInput.t());
            this->backPropagation(rowRealValues.t());
            
            this->gradientDescent(learningRate, rowInput);
            if (j % 1000 == 0)
            {
                float error = this->errCuaMed(result, rowRealValues.t())/10;
                std::cout << "Error: " <<  error << std::endl;
            }
        }
    }

    void trainBatch(float learningRate, const Matrix<T> &inputsBatch, const Matrix<T> &realValuesBatch)
    {
            Matrix<T> result = this->forwardPass(inputsBatch);
            this->backPropagation(realValuesBatch);
            this->gradientDescent(learningRate, inputsBatch);
            
            if (var_print_loss % 150 == 0)
            {
                this->loss = this->errCuaMedBatch(result, realValuesBatch) * 100;
                std::cout << "\tloss: " << this->loss << " %" << std::endl;
            }
            var_print_loss++;
    }

    void show()
    {
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            std::cout << "Layer : " << i << std::endl;
            this->layers[i].show();
        }
    }

    // activation functions
    static T sigmoid(T x) { return 1 / (1 + exp(-x)); }
    static T dSigmoid(T x) { return x * (1 - x); }

    static T relu(T x)
    {
        if (x > 0)
            return x;
        return 0;
    }

    static T dRelu(T x)
    {
        if (x > 0)
            return 1;
        return 0;
    }

    // Error function
    static T errCuaMed(const Matrix<T> &prediction, const Matrix<T> &real)
    {
        Matrix<T> err = prediction - real;
        err.pow(2);
        return Matrix<T>::sum(err) / 2;
    }

    static Matrix<T> dErrCuaMed(const Matrix<T> &prediction, const Matrix<T> &real)
    {
        Matrix<T> dErr = prediction - real;
        return dErr;
    }


    float errCuaMedBatch(const Matrix<T> &pred, const Matrix<T> &real)
    {
        if (pred.rows != real.rows || pred.cols != real.cols)
            throw std::runtime_error("errCuaMedBatch: dimensions must match");

        float sum = 0.0f;

        for (int i = 0; i < pred.rows; i++)
        {
            for (int j = 0; j < pred.cols; j++)
            {
                float diff = pred[i][j] - real[i][j];
                sum += diff * diff;
            }
        }

        // promedio total
        return sum / (pred.rows * pred.cols);
    }

    static float batchLossMSE(const Matrix<float> &pred, const Matrix<float> &real)
    {
        // pred y real: (10, B)
        float total = 0.0f;

        for (int c = 0; c < pred.cols; c++)
        {
            float sampleLoss = 0.0f;

            for (int r = 0; r < pred.rows; r++)
            {
                float diff = pred[r][c] - real[r][c];
                sampleLoss += diff * diff;
            }

            sampleLoss *= 0.5f;
            total += sampleLoss;
        }

        return total / (float)pred.cols; // promedio del batch
    }

    static float batchAccuracy(const Matrix<float> &pred, const Matrix<float> &real)
    {
        // pred y real: (10, B)
        int corrects = 0;

        for (int c = 0; c < pred.cols; c++)
        {
            int p = Matrix<float>::argmaxCol(pred, c);
            int y = Matrix<float>::argmaxCol(real, c);

            if (p == y)
                corrects++;
        }

        return (float)corrects / (float)pred.cols;
    }

    float getLoss(){
    	return this->loss;
    }
};
