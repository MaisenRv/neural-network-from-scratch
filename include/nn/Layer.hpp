#pragma once
#include <cmath>
#include "../Matrix/Matrix.hpp"
#include "../Matrix/MatrixSolverWithThreads.hpp"

template <typename T = float>
class Layer
{
public:
    Matrix<T> weights;
    Matrix<T> bias;
    Matrix<T> weightedSum;
    Matrix<T> activationValues;

    Matrix<T> delta;

    
    Layer(const int numberInputs,const int numberNeurons, int batcheSize) 
        : weights(numberNeurons, numberInputs), bias(numberNeurons, 1), activationValues(numberNeurons,batcheSize),delta(numberNeurons, batcheSize)
    {
        this->weights.randomize();
        this->bias.randomize();
    }

    Matrix<T> calculateLayer(const Matrix<T> &inputs,T (*f)(T), MatrixSolverWithThreads<T> &solver)
    {
        this->weightedSum = solver.sumWithVector(solver.mul(this->weights , inputs), this->bias);
        Matrix<T> activation = this->weightedSum;
        activation.applyFuntion(f);
        this->activationValues = activation;
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
        std::cout << "Deltas" << std::endl;
        this->delta.print();
        std::cout << std::endl;
    }
};
