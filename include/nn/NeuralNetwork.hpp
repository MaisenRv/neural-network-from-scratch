#pragma once

#include <memory>
#include <vector>
#include <nn/Layer/ILayer.hpp>
#include <BitMth/math/Loss.hpp>

namespace NN {
    template <typename T>
    class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers;
        BitMth::Math::LossFunct lossFunctType;

    public:
        NeuralNetwork(BitMth::Math::LossFunct lossFunctType): lossFunctType(lossFunctType){}

        void addLayer(std::unique_ptr<ILayer<T>> layer){
            if(!this->layers.empty()){
                if(this->layers.back()->getNumberNeurons() != layer->getNumberInputs()){
                    throw std::runtime_error(
                        "Error en addLayer: Dimensiones incompatibles. "
                        "La capa anterior expone " + std::to_string(this->layers.back()->getNumberNeurons()) + " salidas, "
                        "pero la nueva capa requiere " + std::to_string(layer->getNumberInputs()) + " entradas."
                    );
                }
            }
            this->layers.push_back(std::move(layer));
        }

        BitMth::Matrix<T> forwardPass(const BitMth::Matrix<T> &inputs){
            BitMth::Matrix<T> result = inputs;
            for (size_t i = 0; i < this->layers.size(); i++){
                result = this->layers[i]->forward(result);
            }
            return result;
        }

        void backPropagation(const BitMth::Matrix<T> &real){
            const auto& predicted = this->layers.back()->getActivationValues();
            BitMth::Matrix<T> errorGradient;

            switch (this->lossFunctType) {
                case BitMth::Math::LossFunct::MSE :
                    errorGradient = BitMth::Math::mseDerivative(predicted, real);
                    break;
                    
                case BitMth::Math::LossFunct::BINARY_CROSS_ENTROPY :
                    errorGradient = BitMth::Math::bceDerivative(predicted, real);
                    break;
            }
            for (size_t i = this->layers.size(); i > 0; i--){
                errorGradient = this->layers[i - 1]->backward(errorGradient);
            }
        }

        void gradientDescent(T learningRate){
            for (size_t i = 0; i < this->layers.size(); i++){
                this->layers[i]->updateParameters(learningRate);   
            }
        }

        void train(T learningRate, const BitMth::Matrix<T> &inputs, const BitMth::Matrix<T> &realValues){
            this->forwardPass(inputs);
            this->backPropagation(realValues);
            this->gradientDescent(learningRate);
        }
    };
}