#pragma once

#include "BitMth/ia/Activations.hpp"
#include <BitMth/random/MatrixRandom.hpp>
#include <nn/Layer/ILayer.hpp>

namespace NN {
    template <typename T>
    class DenseLayer : public ILayer<T>{
    private:
        using Matrix = BitMth::linalg::Matrix<T>;
        using ActFuncType = BitMth::ia::types::ActivationFunctType;

        Matrix weights;
        Matrix bias;

        Matrix dWeights;
        Matrix dBias;

        void selectRandomType(size_t inputs, size_t neurons){
            switch (this->activationFuntType) {
                case ActFuncType::RELU :
                    BitMth::random::heNormal(this->weights, inputs);
                    BitMth::random::uniform(this->bias, T(0.0001), T(0.01));
                    break;
                case ActFuncType::SIGMOID :
                    BitMth::random::xavierUniform(this->weights, inputs, neurons);
                    break;
                case ActFuncType::TANH :
                    BitMth::random::xavierUniform(this->weights, inputs, neurons);
                    break;
                case ActFuncType::SOFTMAX :
                    BitMth::random::xavierUniform(this->weights, inputs, neurons);
                    break;
            }
        }

    public:
        DenseLayer(size_t inputs, size_t neurons, const BitMth::ia::types::ActivationFunctType actFunctType): weights(inputs,neurons), bias(1, neurons){
            this->activationFuntType = actFunctType;
            this->activationFuncts = BitMth::ia::getActivationFunction<T>(this->activationFuntType);
            this->selectRandomType(inputs, neurons);
        };

        ~DenseLayer() = default;

        Matrix forward(const Matrix& input) override {
            this->inputCache = input;
            this->weightedSum = (input * this->weights).addRowVector(this->bias);

            this->activationValues = this->activationFuncts.function(this->weightedSum, nullptr);
            return this->activationValues;
        }

        Matrix backward(const Matrix& errorGradient) override{
            if(this->activationFuncts.devFunction == nullptr){
                this->delta = errorGradient;
            }else{
                this->delta = this->activationFuncts.devFunction(
                    this->weightedSum,
                    this->activationValues,
                    nullptr
                );
                this->delta.hadamardInPlace(errorGradient);
            }
            this->dWeights = Matrix::t(this->inputCache) * this->delta;
            this->dBias = this->delta.reduceSumRows();
            return this->delta * Matrix::t(this->weights);
        }

        void updateParameters(T learningRate) override {
            this->weights -= (learningRate * this->dWeights);
            this->bias    -= (learningRate * this->dBias);
        }

        size_t getNumberNeurons() const override { return this->weights.getCols(); }
        size_t getNumberInputs()  const override { return this->weights.getRows(); }
    };    
} 
