#pragma once

#include <BitMth/random/MatrixRandom.hpp>
#include <nn/Layer/ILayer.hpp>

namespace NN {
    template <typename T>
    class DenseLayer : public ILayer<T>{
    private:
        BitMth::Matrix<T> weights;
        BitMth::Matrix<T> bias;

        BitMth::Matrix<T> dWeights;
        BitMth::Matrix<T> dBias;

        void selectFunctionType(size_t inputs, size_t neurons){
            switch (this->activationFuntType) {
                case BitMth::Math::ActivationFunct::RELU :
                    this->activationFunt = &BitMth::Math::relu;
                    this->derivateActivationFunt = &BitMth::Math::reluDerivative;
                    BitMth::Random::heNormal(this->weights, inputs);
                    BitMth::Random::uniform(this->bias, T(0.0001), T(0.01));
                    break;
                case BitMth::Math::ActivationFunct::SIGMOID :
                    this->activationFunt = &BitMth::Math::sigmoid;
                    this->derivateActivationFunt = &BitMth::Math::sigmoidDerivative;
                    BitMth::Random::xavierUniform(this->weights, inputs, neurons);
                    break;
                case BitMth::Math::ActivationFunct::TANH :
                    this->activationFunt = &BitMth::Math::Tanh;
                    this->derivateActivationFunt = &BitMth::Math::TanhDerivative;
                    BitMth::Random::xavierUniform(this->weights, inputs, neurons);
                    break;
                case BitMth::Math::ActivationFunct::SOFTMAX :
                    this->activationFunt = &BitMth::Math::softmax;
                    BitMth::Random::xavierUniform(this->weights, inputs, neurons);
                    break;
            }
        }

    public:
        DenseLayer(size_t inputs, size_t neurons, BitMth::Math::ActivationFunct  actFunctType): weights(neurons,inputs), bias(neurons,1){
            this->activationFuntType = actFunctType;
            this->selectFunctionType(inputs, neurons);
        };

        ~DenseLayer() = default;

        BitMth::Matrix<T> forward(const BitMth::Matrix<T>& input) override {
            this->inputCache = input;
            this->weightedSum = (this->weights * input) + this->bias;
            this->activationValues = this->activationFunt(this->weightedSum);
            return this->activationValues;
        }

        BitMth::Matrix<T> backward(const BitMth::Matrix<T>& errorGradient) override{
            if(this->activationFuntType == BitMth::Math::ActivationFunct::SOFTMAX){
                this->delta = errorGradient;
            }else{
                this->delta = this->derivateActivationFunt(
                    this->weightedSum,
                    this->activationValues
                );
                this->delta.hadamard(errorGradient);
            }
            this->dWeights = this->delta * this->inputCache.t();
            this->dBias = this->delta.reduceSumCols();
            return this->weights.t() * this->delta;
        }

        void updateParameters(T learningRate) override {
            this->weights -= (learningRate * this->dWeights);
            this->bias    -= (learningRate * this->dBias);
        }

        size_t getNumberNeurons() const override { return this->weights.rows; }
        size_t getNumberInputs()  const override { return this->weights.cols; }
    };    
} 
