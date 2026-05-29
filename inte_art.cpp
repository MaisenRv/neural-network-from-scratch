#include "include/nn/NeuralNetwork.hpp"
#include "include/Matrix/Matrix.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <random>

#define NUM_MIN 0
#define NUM_MAX 40
#define BATCH_SIZE 10

#define SAMPLES_TRAIN_SET 1000 * BATCH_SIZE

#define normalizacion(x) (((x) - NUM_MIN) / (NUM_MAX - NUM_MIN))
#define hip(x,y) (sqrt( (x) * (x) + (y) * (y) ))


int main(){
    std::vector shape = {2,32,16,1};
	
	NeuralNetwork<double> NN(shape, "RELU", BATCH_SIZE);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(NUM_MIN, NUM_MAX);

    for (size_t i = 0; i < 50; i++)
    {
        for (size_t j = 0; j < (SAMPLES_TRAIN_SET / BATCH_SIZE); j++) 
        {
            Matrix<double> X(2, BATCH_SIZE);
            Matrix<double> Y(1, BATCH_SIZE);

            for (size_t k = 0; k < BATCH_SIZE; k++)
            {
                double x = distribution(generator);
                double y = distribution(generator);

                X[0][k] = normalizacion(x);
                X[1][k] = normalizacion(y);

                Y[0][k] = hip(x,y); 
            }

            if(NN.getLoss() > 100){ 
                NN.trainBatch(0.01, X, Y); 
            }
            else if(NN.getLoss() > 1){ 
                NN.trainBatch(0.005, X, Y); 
            }
            else if(NN.getLoss() <= 1 && NN.getLoss() >= 0.5){ 
                NN.trainBatch(0.001, X, Y); 
            }
            else if(NN.getLoss() < 0.5){ 
                NN.trainBatch(0.0005, X, Y); 
            }
        }

        std::cout << "Epoch: " << i + 1 << std::endl;
    }


    while (true)
    {
        double x = 0;
        double y = 0;
        std::cout << "X: ";
        std::cin >> x;

        std::cout << "Y: ";
        std::cin >> y;
        Matrix<double> X_input(2, BATCH_SIZE);
        for (size_t i = 0; i < BATCH_SIZE; i++)
        {
            X_input[0][i] = normalizacion(x);
            X_input[1][i] = normalizacion(y);

        }

        Matrix<double> result = NN.forwardPass(X_input);
        std::cout << " Prediccion: " << result[0][0] << std::endl;
        std::cout << " Real:       " << hip(x,y) << std::endl;
        std::cout << " Error:      " << std::fabs(hip(x,y) - result[0][0]) << "\n"<< std::endl; 
    }
    


    return 0;
}
