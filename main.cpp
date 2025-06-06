#include "include/nn/Layer.hpp"
#include "include/nn/NeuralNetwork.hpp"
#include "include/Matrix/Matrix.hpp"
#include <vector>

int main(){

    Matrix<float> X = {{2,3,6,7}};


    std::vector<int> shape = {4,10,2,1};
    NeuralNetwork NN(shape,"sigmoid");

    NN.forwardPass(X.t());
    NN.show();

    return 0;
}