#pragma once

#include <BitMth/linalg/Matrix.hpp>
#include <BitMth/ia/Activations.hpp>
#include <BitMth/ia/types/ActivationTypes.hpp>

namespace NN {
    template<typename T> 
    class ILayer {
    protected:
        using Matrix = BitMth::linalg::Matrix<T>;

        Matrix weightedSum;        
        Matrix activationValues;
        Matrix delta;
        Matrix inputCache;

        BitMth::ia::types::ActivationContent<T> activationFuncts;
        BitMth::ia::types::ActivationFunctType activationFuntType;
    public:
        ILayer() = default;
        virtual ~ILayer() = default;
        
        virtual Matrix forward(const Matrix& input) = 0;
        virtual Matrix backward(const Matrix& errorGradient) = 0;
        virtual void updateParameters(T learningRate) = 0;

        virtual size_t getNumberNeurons() const = 0;
        virtual size_t getNumberInputs() const = 0;

        const Matrix& getActivationValues() const { return this->activationValues; };
    };    
}
