#pragma once

#include <nn/Layer/types/LayerTypes.hpp>
#include <BitMth/ia/Activations.hpp>

namespace NN::layer::types{
    struct LayerShape{
        size_t input;
        size_t output;
        NN::layer::types::LayerT layerType;
        BitMth::ia::ActivationFunct actFunct;
    };
    
} 