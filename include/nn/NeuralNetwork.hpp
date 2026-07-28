#pragma once

#include "BitMth/ia/types/LossTypes.hpp"
#include <memory>
#include <vector>
#include <nn/Layer/ILayer.hpp>
#include <BitMth/ia/Loss.hpp>

namespace NN {
    template <typename T>
    class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers;

        BitMth::ia::types::LossContent<T> lossFunctions;

    public:
        NeuralNetwork(BitMth::ia::types::LossFunctType lossFunctType):
            lossFunctions(BitMth::ia::getLossFunction<T>(lossFunctType)){}

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

        BitMth::linalg::Matrix<T> forwardPass(const BitMth::linalg::Matrix<T> &inputs){
            BitMth::linalg::Matrix<T> result = inputs;
            for (size_t i = 0; i < this->layers.size(); i++){
                result = this->layers[i]->forward(result);
            }
            return result;
        }

        void backPropagation(const BitMth::linalg::Matrix<T> &real){
            const auto& predicted = this->layers.back()->getActivationValues();
            BitMth::linalg::Matrix<T> errorGradient = this->lossFunctions.lossFunctDev(predicted, real, nullptr);
            for (size_t i = this->layers.size(); i > 0; i--){
                errorGradient = this->layers[i - 1]->backward(errorGradient);
            }
        }

        void gradientDescent(T learningRate){
            for (size_t i = 0; i < this->layers.size(); i++){
                this->layers[i]->updateParameters(learningRate);   
            }
        }

        void train(T learningRate, const BitMth::linalg::Matrix<T> &inputs, const BitMth::linalg::Matrix<T> &realValues){
            (void)this->forwardPass(inputs);
            this->backPropagation(realValues);
            this->gradientDescent(learningRate);
        }
    };
}
