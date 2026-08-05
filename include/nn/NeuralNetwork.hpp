#pragma once

#include <BitMth/core/Arena.hpp>
#include <chrono>
#include <cstddef>
#include <memory>
#include <vector>
#include <nn/Layer/ILayer.hpp>
#include <BitMth/ia/Loss.hpp>
#include <BitMth/ia/types/OptimizerTypes.hpp>
#include <nn/Layer/types/LayerShape.hpp>

namespace NN {
    template <typename T>
    class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers;
        BitMth::ia::types::LossContent<T> lossFunctions;
        BitMth::ia::types::OptimizerType optType; 

        std::unique_ptr<BitMth::core::Arena> staticArena;
    public:
        NeuralNetwork(BitMth::ia::types::LossFunctType lossFunctType, BitMth::ia::types::OptimizerType optType):
            lossFunctions(BitMth::ia::getLossFunction<T>(lossFunctType)), optType(optType){}

        void setLayers(const std::vector<layer::types::LayerShape> shape){
            for (size_t i = 0; i < shape.size() - 1; i++) {
                if (shape[i].output != shape[i + 1].input) {
                    throw std::runtime_error(
                        "Error en setLayer: Dimensiones incompatibles. "
                        "La capa anterior expone " + std::to_string(shape[i].output) + " salidas, "
                        "pero la nueva capa requiere " + std::to_string(shape[i + 1].input) + " entradas."
                    );
                }            
            }

            size_t bytesArena = 0;
            for(const auto& layerShape: shape){
                bytesArena += (layerShape.output * sizeof(T))* (layerShape.input + 1); 
            }
            bytesArena += size_t(float(bytesArena) * 0.1);
            staticArena = std::make_unique<BitMth::core::Arena>(bytesArena);
            
            layers.reserve(shape.size());
            for(const auto& layerShape: shape){
                layers.emplace_back(layer::types::getLayerType<T>(layerShape,staticArena.get()));
            }
        }

        BitMth::linalg::Matrix<T> forwardPass(const BitMth::linalg::Matrix<T> &inputs, BitMth::core::Arena * arena = nullptr){
            BitMth::linalg::Matrix<T> result = inputs;
            for (size_t i = 0; i < this->layers.size(); i++){
                result = this->layers[i]->forward(result, arena);
            }
            return result;
        }

        void backPropagation(const BitMth::linalg::Matrix<T> &real, BitMth::core::Arena * arena = nullptr){
            const auto& predicted = this->layers.back()->getActivationValues();
            BitMth::linalg::Matrix<T> errorGradient = this->lossFunctions.lossFunctDev(predicted, real, nullptr);
            for (size_t i = this->layers.size(); i > 0; i--){
                errorGradient = this->layers[i - 1]->backward(errorGradient,arena);
            }
        }

        void gradientDescent(T learningRate, BitMth::core::Arena * arena = nullptr){
            for (size_t i = 0; i < this->layers.size(); i++){
                this->layers[i]->updateParameters(learningRate,optType,arena);   
            }
        }

        void train(T learningRate, const BitMth::linalg::Matrix<T> &inputs, const BitMth::linalg::Matrix<T> &realValues, BitMth::core::Arena * arena = nullptr){
            (void)this->forwardPass(inputs,arena);
            this->backPropagation(realValues,arena);
            this->gradientDescent(learningRate,arena);
        }
    };
}
