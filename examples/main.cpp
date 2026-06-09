// #include "nn/NeuralNetwork.hpp"
// #include <vector>
// #include <iostream>
#include "utils/readDataSet.hpp"
// #include "BitMth/math/Activations.hpp"

#include <nn/NeuralNetwork.hpp>
#include <nn/Layer/DenseLayer.hpp>
int main()
{
	using BitMth::ia::ActivationFunct;
    using BitMth::ia::LossFunct;
    using BitMth::linalg::Matrix;
    
    // 1. CARGA DEL DATASET DE ENTRENAMIENTO (60,000 imágenes)
    ReadDataSet data("datasets/train-images-idx3-ubyte", true);
    ReadDataSet dataLabels("datasets/train-labels-idx1-ubyte", false);
    int dataLength = data.width * data.height; // 784

    // 2. CONFIGURACIÓN DE LA RED NEURONAL
    NN::NeuralNetwork<float> nn(LossFunct::MSE);
    nn.addLayer(std::make_unique<NN::DenseLayer<float>>(dataLength, 128, ActivationFunct::RELU));
    nn.addLayer(std::make_unique<NN::DenseLayer<float>>(128, 64, ActivationFunct::RELU));
    nn.addLayer(std::make_unique<NN::DenseLayer<float>>(64, 10, ActivationFunct::SIGMOID));

    std::cout << "Cargando dataset de entrenamiento en matrices de BitMth..." << std::endl;

    // Matriz de características: (60000, 784)
    Matrix<float> X(data.samples, dataLength);
    for (size_t i = 0; i < data.samples; i++) {
        unsigned char* image = data.readNextImage();
        for (size_t j = 0; j < dataLength; j++) {
            X(i, j) = image[j] / 255.0f; // Normalización elemental
        }    
    }

    // Matriz de etiquetas One-Hot: (60000, 10)
    Matrix<float> Y(dataLabels.samples, 10);
    for (size_t i = 0; i < dataLabels.samples; i++) {
        unsigned char label;
        dataLabels.readNextLabel(label);
        for (size_t j = 0; j < 10; j++) {
            Y(i, j) = (label == j) ? 1.0f : 0.0f;
        }    
    }

    // 3. BUCLE DE ENTRENAMIENTO EN MINI-BATCHES (Estabilidad Numérica)
    int epochs = 5;
    int batchSize = 64; 
    float learningRate = 0.01f;

    std::cout << "Iniciando entrenamiento..." << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epochLoss = 0.0f;
        int numBatches = 0;

        for (size_t start = 0; start < data.samples; start += batchSize) {
            size_t currentBatchSize = std::min(static_cast<size_t>(batchSize), data.samples - start);

            Matrix<float> XBatch(currentBatchSize, dataLength);
            Matrix<float> YBatch(currentBatchSize, 10);

            for (size_t i = 0; i < currentBatchSize; i++) {
                for (size_t j = 0; j < dataLength; j++) {
                    XBatch(i, j) = X(start + i, j);
                }
                for (size_t j = 0; j < 10; j++) {
                    YBatch(i, j) = Y(start + i, j);
                }
            }

            nn.train(learningRate, XBatch, YBatch);

            Matrix<float> prediction = nn.forwardPass(XBatch);
            
            epochLoss += BitMth::ia::mse(prediction, YBatch);
            numBatches++;
        }

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                  << " | Avg Loss: " << (epochLoss / numBatches) << std::endl;
    }

    // 4. VALIDACIÓN CON EL DATASET DE TEST (10,000 imágenes para evaluar el Accuracy real)
    std::cout << "\nCargando dataset de validación (Test)..." << std::endl;
    ReadDataSet dataTest("datasets/t10k-images-idx3-ubyte", true);
    ReadDataSet dataLabelsTest("datasets/t10k-labels-idx1-ubyte", false);

    int corrects = 0;

    for (size_t i = 0; i < dataTest.samples; i++) {
        Matrix<float> xSample(1, dataLength);
        unsigned char* image = dataTest.readNextImage();
        for (size_t j = 0; j < dataLength; j++) {
            xSample(0, j) = image[j] / 255.0f;
        }

        unsigned char actualLabel;
        dataLabelsTest.readNextLabel(actualLabel);

        Matrix<float> result = nn.forwardPass(xSample);

        int highestIndex = 0;
        float maxVal = result(0, 0);
        for (int j = 1; j < 10; j++) {
            if (result(0, j) > maxVal) {
                maxVal = result(0, j);
                highestIndex = j;
            }
        }

        if (highestIndex == actualLabel) {
            corrects++;
        }
    }

    std::cout << "\n--- RESULTADO DE LA EVALUACIÓN ---" << std::endl;
    std::cout << "Muestras correctas: " << corrects << " / " << dataTest.samples << std::endl;
    std::cout << "Precisión Final (Accuracy): " << (static_cast<float>(corrects) / dataTest.samples) * 100.0f << "%" << std::endl;

    return 0;
}
