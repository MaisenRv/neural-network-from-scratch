#pragma once
#include "Layer.hpp"
#include <string>

template <typename T = float>
class NeuralNetwork
{
public:
    std::vector<int> shape;
    std::vector<Layer<T>> layers;
    

    float learningRate;

    T (*activationFunction)(T);
    T (*derivativeActivationFunction)(T);

    NeuralNetwork(std::vector<int> &shape, std::string a_func)
    {
        this->activationFunction = NeuralNetwork<T>::sigmoid;
        if (a_func == "sigmoid")
            this->activationFunction = NeuralNetwork<T>::sigmoid;
        if (a_func == "RELU")
            this->activationFunction = NeuralNetwork<T>::relu;

        this->shape = shape;

        for (size_t i = 0; i < this->shape.size() - 1; i++)
        {
            this->layers.push_back(Layer<T>(this->shape[i], this->shape[i + 1]));
        }
    }

    void forwardPass(const Matrix<T> &inputs)
    {
        Matrix<T> result;
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            if (i == 0)
            {
                result = this->layers[i].calculateLayer(inputs, NeuralNetwork<T>::sigmoid);
                continue;
            }
            result = this->layers[i].calculateLayer(result, NeuralNetwork<T>::sigmoid);
        }
    }

    void show(){
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
        return Matrix<T>::mean(err);
    }
};
