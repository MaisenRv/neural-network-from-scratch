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
    Matrix<T> activationValues;

    
    Layer(const int numberInputs,const int numberNeurons) 
        : weights(numberNeurons, numberInputs), bias(numberNeurons, 1), activationValues(numberNeurons,1)
    {
        this->weights.randomize();
        this->bias.randomize();
    }

    Matrix<T> calculateLayer(const Matrix<T> &inputs,T (*f)(T))
    {
        Matrix<T> weightedSum = ((this->weights * inputs) + this->bias);
        weightedSum.applyFuntion(f);
        this->activationValues = weightedSum;
        return this->activationValues;
    }

    void show()
    {
        std::cout << "Weights" << std::endl;
        this->weights.print();
        std::cout << "Bias" << std::endl;
        this->bias.print();
        std::cout << "Activation values" << std::endl;
        this->activationValues.print();
        std::cout << std::endl;
    }
};
