#pragma once

#include "BitMth/core/Arena.hpp"
#include <memory>
#include <nn/Layer/types/LayerTypes.hpp>
#include <BitMth/ia/types/ActivationTypes.hpp>
#include <nn/Layer/ILayer.hpp>
#include <nn/Layer/DenseLayer.hpp>

namespace NN::layer::types{
    struct LayerShape{
        size_t input;
        size_t output;
        LayerT layerType;
        BitMth::ia::types::ActivationFunctType actFunct;
    };
    template<typename T>
    inline std::unique_ptr<ILayer<T>> getLayerType(const LayerShape& layerShape, BitMth::core::Arena *staticArena){
        switch (layerShape.layerType) {
            case LayerT::DENSE:
                return 
                    std::make_unique<NN::DenseLayer<T>>(
                        layerShape.input,
                        layerShape.output,
                        layerShape.actFunct,
                        staticArena
                    );
            break;
            default:
                return 
                    std::make_unique<NN::DenseLayer<T>>(
                        layerShape.input,
                        layerShape.output,
                        layerShape.actFunct,
                        staticArena
                    );
        }
    }
} 
