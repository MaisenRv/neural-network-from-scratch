#pragma once

#include <BitMth/random/MatrixRandom.hpp>
#include <nn/Layer/ILayer.hpp>

namespace NN {
    template <typename T>
    class DenseLayer : public ILayer<T>{
    private:
        Bitmth::Matrix<T> weights;
        Bitmth::Matrix<T> bias;

        Bitmth::Matrix<T> dWeights;
        Bitmth::Matrix<T> dBias;

        void selectFunctionType(size_t inputs, size_t neurons){
            switch (this->activationFuntType) {
                case Bitmth::Math::ActivationFunct::RELU :
                    this->activationFunt = &Bitmth::Math::relu;
                    this->derivateActivationFunt = &Bitmth::Math::reluDerivative;
                    Bitmth::Random::heNormal(this->weights, inputs);
                    Bitmth::Random::uniform(this->bias, T(0.0001), T(0.01));
                    break;
                case Bitmth::Math::ActivationFunct::SIGMOID :
                    this->activationFunt = &Bitmth::Math::sigmoid;
                    this->derivateActivationFunt = &Bitmth::Math::sigmoidDerivative;
                    Bitmth::Random::xavierUniform(this->weights, inputs, neurons);
                    break;
                case Bitmth::Math::ActivationFunct::TANH :
                    this->activationFunt = &Bitmth::Math::Tanh;
                    this->derivateActivationFunt = &Bitmth::Math::TanhDerivative;
                    Bitmth::Random::xavierUniform(this->weights, inputs, neurons);
                    break;
                case Bitmth::Math::ActivationFunct::SOFTMAX :
                    this->activationFunt = &Bitmth::Math::softmax;
                    Bitmth::Random::xavierUniform(this->weights, inputs, neurons);
                    break;
            }
        }

    public:
        DenseLayer(size_t inputs, size_t neurons, Bitmth::Math::ActivationFunct  actFunctType): weights(neurons,inputs), bias(neurons,1){
            this->activationFuntType = actFunctType;
            this->selectFunctionType(inputs, neurons);
        };

        ~DenseLayer() = default;

        Bitmth::Matrix<T> forward(const Bitmth::Matrix<T>& input) override {
            this->inputCache = input;
            this->weightedSum = (this->weights * input) + this->bias;
            this->activationValues = this->activationFunt(this->weightedSum);
        }

        Bitmth::Matrix<T> backward(const Bitmth::Matrix<T>& errorGradient) override{
            if(this->activationFuntType == Bitmth::Math::ActivationFunct::SOFTMAX){
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

        
    };    
} 
