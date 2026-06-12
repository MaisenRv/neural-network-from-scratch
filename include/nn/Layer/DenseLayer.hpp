#pragma once

#include <BitMth/random/MatrixRandom.hpp>
#include <nn/Layer/ILayer.hpp>

namespace NN {
    template <typename T>
    class DenseLayer : public ILayer<T>{
    private:
        BitMth::linalg::Matrix<T> weights;
        BitMth::linalg::Matrix<T> bias;

        BitMth::linalg::Matrix<T> dWeights;
        BitMth::linalg::Matrix<T> dBias;

        void selectFunctionType(size_t inputs, size_t neurons){
            switch (this->activationFuntType) {
                case BitMth::ia::ActivationFunct::RELU :
                    this->activationFunt = &BitMth::ia::relu;
                    this->derivateActivationFunt = &BitMth::ia::reluDerivative;
                    BitMth::random::heNormal(this->weights, inputs);
                    BitMth::random::uniform(this->bias, T(0.0001), T(0.01));
                    break;
                case BitMth::ia::ActivationFunct::SIGMOID :
                    this->activationFunt = &BitMth::ia::sigmoid;
                    this->derivateActivationFunt = &BitMth::ia::sigmoidDerivative;
                    BitMth::random::xavierUniform(this->weights, inputs, neurons);
                    break;
                case BitMth::ia::ActivationFunct::TANH :
                    this->activationFunt = &BitMth::ia::Tanh;
                    this->derivateActivationFunt = &BitMth::ia::TanhDerivative;
                    BitMth::random::xavierUniform(this->weights, inputs, neurons);
                    break;
                case BitMth::ia::ActivationFunct::SOFTMAX :
                    this->activationFunt = &BitMth::ia::softmax;
                    BitMth::random::xavierUniform(this->weights, inputs, neurons);
                    break;
            }
        }

    public:
        DenseLayer(size_t inputs, size_t neurons, BitMth::ia::ActivationFunct  actFunctType): weights(inputs,neurons), bias(1, neurons){
            this->activationFuntType = actFunctType;
            this->selectFunctionType(inputs, neurons);
        };

        ~DenseLayer() = default;

        BitMth::linalg::Matrix<T> forward(const BitMth::linalg::Matrix<T>& input) override {
            this->inputCache = input;
            this->weightedSum = (input * this->weights).addRowVector(this->bias);

            this->activationValues = this->activationFunt(this->weightedSum);
            return this->activationValues;
        }

        BitMth::linalg::Matrix<T> backward(const BitMth::linalg::Matrix<T>& errorGradient) override{
            if(this->activationFuntType == BitMth::ia::ActivationFunct::SOFTMAX){
                this->delta = errorGradient;
            }else{
                this->delta = this->derivateActivationFunt(
                    this->weightedSum,
                    this->activationValues
                );
                this->delta.hadamard(errorGradient);
            }
            this->dWeights = this->inputCache.t() * this->delta;
            this->dBias = this->delta.reduceSumRows();
            return this->delta * this->weights.t();
        }

        void updateParameters(T learningRate) override {
            this->weights -= (learningRate * this->dWeights);
            this->bias    -= (learningRate * this->dBias);
        }

        size_t getNumberNeurons() const override { return this->weights.cols; }
        size_t getNumberInputs()  const override { return this->weights.rows; }
    };    
} 
