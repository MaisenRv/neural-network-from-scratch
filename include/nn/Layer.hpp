#pragma once
#include <cmath>
#include "../Matrix/Matrix.hpp"
// #include "Neuron.hpp"

template <typename T = float>
class Layer
{
public:
    Matrix<T> weights;
    Matrix<T> bias;
    // activation functions
    static T sigmoid(T x) { return 1 / (1 + exp(-x)); }
    static T max(T x)
    {
        if (x > 0)
        {
            return x;
        }
        return 0;
    }

    Layer(const int numberInputs,const int numberNeurons) : weights(numberNeurons, numberInputs), bias(numberNeurons, 1)
    {
        this->weights.randomize();
        this->bias.randomize();
    }

    Matrix<T> calculateLayer(const Matrix<T> &inputs)
    {
        Matrix<T> weightedSum = ((this->weights * inputs.t()) + this->bias).t();
        weightedSum.applyFuntion(Layer<T>::sigmoid);
        weightedSum.print();
        return weightedSum;
    }

    void show()
    {
        std::cout << "weights" << std::endl;
        this->weights.print();
        std::cout << "bias" << std::endl;
        this->bias.print();
        std::cout << std::endl;
    }
};
