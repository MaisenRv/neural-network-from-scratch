#pragma once
#include "Layer.hpp"

template <typename T = float>
class NeuralNetwork{
public:
    std::vector<int> shape;
    std::vector<Layer<T>> layers;

    NeuralNetwork(std::vector<int> &shape){
        this->shape = shape;
        for (size_t i = 0; i < this->shape.size() - 1 ; i++)
        {
            this->layers.push_back(Layer<T>(this->shape[i],this->shape[i + 1]));
        }   
    }
    void cal(const Matrix<T> &inputs){
        Matrix<T> result;
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            std::cout << "Iteracion " << i << std::endl;
            if (i == 0){
                result = this->layers[i].calculateLayer(inputs);
                continue;
            }
            result = this->layers[i].calculateLayer(result);
        }
        std::cout << "Resultado " << std::endl;
        result.print();
    }

    void show(){
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            std::cout << "Layer : " << i << std::endl;
            this->layers[i].show();
        }
        
    }
};

