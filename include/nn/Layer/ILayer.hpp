#pragma once

#include <BitMth/core/Matrix.hpp>
#include <BitMth/math/Activations.hpp>

namespace NN {
    template<typename T> 
    class ILayer {
    protected:
        using ActivationPtr = Bitmth::Matrix<T>(*)(const Bitmth::Matrix<T>&);
        using DerivativePtr = Bitmth::Matrix<T>(*)(const Bitmth::Matrix<T>&, const Bitmth::Matrix<T>&);
        
        Bitmth::Matrix<T> weightedSum;        
        Bitmth::Matrix<T> activationValues;
        Bitmth::Matrix<T> delta;

        Bitmth::Math::ActivationFunct activationFuntType;
        ActivationPtr activationFunt;
        DerivativePtr derivateActivationFunt;
    public:
        ILayer() = default;
        virtual ~ILayer() = default;
        
        virtual Bitmth::Matrix<T> forward(const Bitmth::Matrix<T>& input) = 0;
        virtual Bitmth::Matrix<T> backward(const Bitmth::Matrix<T>& errorGradient) = 0;
    };    
}
