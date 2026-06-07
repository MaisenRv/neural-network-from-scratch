#pragma once

#include <BitMth/linalg/Matrix.hpp>
#include <BitMth/ia/Activations.hpp>

namespace NN {
    template<typename T> 
    class ILayer {
    protected:
        using ActivationPtr = BitMth::linalg::Matrix<T>(*)(const BitMth::linalg::Matrix<T>&);
        using DerivativePtr = BitMth::linalg::Matrix<T>(*)(const BitMth::linalg::Matrix<T>&, const BitMth::linalg::Matrix<T>&);
        
        BitMth::linalg::Matrix<T> weightedSum;        
        BitMth::linalg::Matrix<T> activationValues;
        BitMth::linalg::Matrix<T> delta;
        BitMth::linalg::Matrix<T> inputCache;

        BitMth::ia::ActivationFunct activationFuntType;
        ActivationPtr activationFunt;
        DerivativePtr derivateActivationFunt;
    public:
        ILayer() = default;
        virtual ~ILayer() = default;
        
        virtual BitMth::linalg::Matrix<T> forward(const BitMth::linalg::Matrix<T>& input) = 0;
        virtual BitMth::linalg::Matrix<T> backward(const BitMth::linalg::Matrix<T>& errorGradient) = 0;
        virtual void updateParameters(T learningRate) = 0;

        virtual size_t getNumberNeurons() const = 0;
        virtual size_t getNumberInputs() const = 0;

        const BitMth::linalg::Matrix<T>& getActivationValues() const { return this->activationValues; };
    };    
}
