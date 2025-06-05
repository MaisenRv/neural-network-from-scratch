#include "include/nn/Layer.hpp"
#include "include/nn/NeuralNetwork.hpp"
#include "include/Matrix/Matrix.hpp"
#include <vector>

int main(){
    // Layer n(10,20);
    // n.show();
    Matrix<float> X = {{2,3,3,4,5,6,7,8,9,6}};
    // n.calculateLayer(X);
    std::vector<int> shape = {10,4,8,4,1};
    NeuralNetwork NN(shape);

    NN.cal(X);
   
    return 0;
}