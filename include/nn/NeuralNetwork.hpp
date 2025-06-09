#pragma once
#include "Layer.hpp"
#include <string>

template <typename T = float>
class NeuralNetwork
{
public:
    std::vector<int> shape;
    std::vector<Layer<T>> layers;

    T (*activationFunction)(T);
    T (*derivativeActivationFunction)(T);

    NeuralNetwork(std::vector<int> &shape, std::string a_func)
    {
        this->activationFunction = NeuralNetwork<T>::sigmoid;
        if (a_func == "sigmoid"){
            this->activationFunction = NeuralNetwork<T>::sigmoid;
            this->derivativeActivationFunction = NeuralNetwork<T>::dSigmoid;
        }
        if (a_func == "RELU"){
            this->activationFunction = NeuralNetwork<T>::relu;
            this->derivativeActivationFunction = NeuralNetwork<T>::dRelu;
        }
        this->shape = shape;

        for (size_t i = 0; i < this->shape.size() - 1; i++)
        {
            this->layers.push_back(Layer<T>(this->shape[i], this->shape[i + 1]));            
        }
    }

    Matrix<T> forwardPass(const Matrix<T> &inputs)
    {
        Matrix<T> result;
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            if (i == 0)
            {
                result = this->layers[i].calculateLayer(inputs, this->activationFunction);
                continue;
            }
            result = this->layers[i].calculateLayer(result, this->activationFunction);
        }
        return result;
    }

    void backPropagation( const Matrix<T> &real){
        // dC/dW = dC/dA * dA/dz * dZ/dW  -----  dZ/dW = a^L-1
        // dC/dB = dC/dA * dA/dz * dZ/dB  -----  dZ/dB = 1  


        // delta^L = dC/dA^L * dA^L/dz^L
        // dC/dW = delta^L * dZ/dW  -----  dZ/dW = a^L-1
        // dC/dB = delta^L * dZ/dB  -----  dZ/dB = 1

        for (int i = this->layers.size() - 1; i >= 0; i--)
        {
            // a^L
            Matrix<T> aValuesLayer = this->layers[i].activationValues;
            if(this->layers.size() - 1 == i ){
                // dC/dA
                Matrix<T> dC = NeuralNetwork<T>::dErrCuaMed(aValuesLayer,real);

                // dA/dz
                Matrix<T> dA = aValuesLayer;
                dA.applyFuntion(this->derivativeActivationFunction);
                dC.hadamardProduct(dA);

                // Delta
                this->layers[i].delta = dC;
                continue;
            }
            
            // delta^L = (delta^L+1 * W^L+1) *  dA^L/dz^L
            Matrix<T> deltaLayerL = this->layers[i+1].delta.t() * this->layers[i+1].weights;
            // dA^L/dz^L
            Matrix<T> dA = aValuesLayer;
            dA.applyFuntion(this->derivativeActivationFunction);
        
            deltaLayerL.hadamardProduct(dA);
            this->layers[i].delta = deltaLayerL.t();
        }
        
    }

    void gradientDescent(float learningRate,const Matrix<T> &inputs){
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            this->layers[i].bias = this->layers[i].bias - (this->layers[i].delta * learningRate);
            if(i==0){
                this->layers[i].weights = this->layers[i].weights - ((this->layers[i].delta *  inputs) * learningRate);
                continue;
            }
            this->layers[i].weights = this->layers[i].weights - ((this->layers[i - 1].activationValues * this->layers[i].delta.t()) * learningRate).t();
        }
    }

    void train(float learningRate,const Matrix<T> &inputs,const Matrix<T> &realValues){
        for (size_t i = 0; i < 10000; i++)
        {
            for (size_t i = 0; i < inputs.rows; i++)
            {
                Matrix<T> rowInput = inputs.getRow(i);
                Matrix<T> rowRealValues = realValues.getRow(i);
                
                Matrix<T>  result = this->forwardPass(rowInput.t());
                this->backPropagation(rowRealValues.t());
                this->gradientDescent(learningRate,rowInput);
                std::cout << "Error: " << this->errCuaMed(result,rowRealValues) << std::endl;
            }
        }
    }


    void show(){
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            std::cout << "Layer : " << i << std::endl;
            this->layers[i].show();
        }
        
    }

        // activation functions
    static T sigmoid(T x) { return 1 / (1 + exp(-x)); }
    static T dSigmoid(T x) { return x * (1 - x); }

    static T relu(T x)
    {
        if (x > 0)
            return x;
        return 0;
    }

    static T dRelu(T x)
    {
        if (x > 0)
            return 1;
        return 0;
    }

    // Error function
    static T errCuaMed(const Matrix<T> &prediction, const Matrix<T> &real)
    {
        Matrix<T> err = prediction - real;
        err.pow(2);
        return Matrix<T>::sum(err)/2;
    }

    static Matrix<T> dErrCuaMed(const Matrix<T> &prediction, const Matrix<T> &real)
    {
        Matrix<T> dErr = prediction - real;
        return dErr;
    }
};
