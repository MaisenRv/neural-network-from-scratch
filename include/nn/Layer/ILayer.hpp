#pragma once

#include <BitMth/core/Matrix.hpp>
#include <BitMth/math/Activations.hpp>

namespace NN {
    template<typename T> 
    class ILayer {
    protected:
        using ActivationPtr = BitMth::Matrix<T>(*)(const BitMth::Matrix<T>&);
        using DerivativePtr = BitMth::Matrix<T>(*)(const BitMth::Matrix<T>&, const BitMth::Matrix<T>&);
        
        BitMth::Matrix<T> weightedSum;        
        BitMth::Matrix<T> activationValues;
        BitMth::Matrix<T> delta;
        BitMth::Matrix<T> inputCache;

        BitMth::Math::ActivationFunct activationFuntType;
        ActivationPtr activationFunt;
        DerivativePtr derivateActivationFunt;
    public:
        ILayer() = default;
        virtual ~ILayer() = default;
        
        virtual BitMth::Matrix<T> forward(const BitMth::Matrix<T>& input) = 0;
        virtual BitMth::Matrix<T> backward(const BitMth::Matrix<T>& errorGradient) = 0;
        virtual void updateParameters(T learningRate) = 0;

        virtual size_t getNumberNeurons() const = 0;
        virtual size_t getNumberInputs() const = 0;

        const BitMth::Matrix<T>& getActivationValues() const { return this->activationValues; };
    };    
}
