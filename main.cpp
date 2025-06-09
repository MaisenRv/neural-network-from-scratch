#include "include/nn/Layer.hpp"
#include "include/nn/NeuralNetwork.hpp"
#include "include/Matrix/Matrix.hpp"
#include <vector>

int main(){

    // Matrix<float> X = {{1,1,1,1,1}};
    // Matrix<float> Y = {{1}};
    // Matrix<float> X = {{0.9, 0.1, 0.3, 0.7, 0.2}};
    // Matrix<float> Y = {{1}};

    Matrix<double> X = {{0,0},{0,1},{1,0},{1,1}};
    Matrix<double> Y = {{0},{1},{1},{0}};

    Matrix<double> P = {{0,0}};
    Matrix<double> P1 = {{0,1}};
    Matrix<double> P2 = {{1,0}};
    Matrix<double> P3 = {{1,1}};
    std::vector<int> shape = {2,2,1};
    NeuralNetwork<double> NN(shape,"sigmoid");
      
    NN.train(1,X,Y);

    Matrix result = NN.forwardPass(P.t());
    Matrix result1 = NN.forwardPass(P1.t());
    Matrix result2 = NN.forwardPass(P2.t());
    Matrix result3 = NN.forwardPass(P3.t());

    std::cout << std::endl;
    result.print();
    result1.print();
    result2.print();
    result3.print();

    return 0;
}