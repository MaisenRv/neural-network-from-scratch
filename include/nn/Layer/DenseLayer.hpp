#pragma once

#include "BitMth/core/Arena.hpp"
#include "BitMth/linalg/Matrix.hpp"
#include <BitMth/random/MatrixRandom.hpp>
#include <nn/Layer/ILayer.hpp>
#include <BitMth/utils/Constants.hpp>

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
        DenseLayer(size_t inputs, size_t neurons, const ActFuncType actFunctType, BitMth::core::Arena *staticArena): weights(inputs,neurons,staticArena), bias(1, neurons,staticArena){
            this->activationFuntType = actFunctType;
            this->activationFuncts = BitMth::ia::getActivationFunction<T>(this->activationFuntType);
            this->selectRandomType(inputs, neurons);
        };

        ~DenseLayer() = default;

        Matrix forward(const Matrix& input, BitMth::core::Arena * arena = nullptr) override {
            this->inputCache = input;
            // this->weightedSum = (input * this->weights).addRowVector(this->bias);
            this->weightedSum = Matrix::mul(input, this->weights, arena).addRowVector(this->bias,arena);

            this->activationValues = this->activationFuncts.function(this->weightedSum, arena);
            return this->activationValues;
        }

        Matrix backward(const Matrix& errorGradient, BitMth::core::Arena * arena = nullptr) override{
            if(this->activationFuncts.devFunction == nullptr){
                this->delta = errorGradient;
            }else{
                this->delta = this->activationFuncts.devFunction(
                    this->weightedSum,
                    this->activationValues,
                    arena
                );
                this->delta.hadamardInPlace(errorGradient);
            }
            // this->dWeights = Matrix::t(this->inputCache, arena) * this->delta;
            this->dWeights = Matrix::mul(Matrix::t(this->inputCache, arena) , this->delta,arena);
            this->dBias = this->delta.reduceSumRows(arena);
            return this->delta * Matrix::t(this->weights,arena);
        }

        void updateParameters(T learningRate, BitMth::ia::types::OptimizerType type, BitMth::core::Arena * arena = nullptr) override {
            auto& optimize = BitMth::ia::getOptimizer<T>(type);
            optimize.opt(this->weights, this->dWeights, learningRate, this->stateOptWeight, BitMth::utils::WEIGHT_DECAY<T>);
            optimize.opt(this->bias, this->dBias, learningRate, this->stateOptBias, BitMth::utils::WEIGHT_DECAY<T>);
        }

        size_t getNumberNeurons() const override { return this->weights.getCols(); }
        size_t getNumberInputs()  const override { return this->weights.getRows(); }
    };    
} 
