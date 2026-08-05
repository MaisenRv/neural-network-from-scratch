#include "BitMth/core/Arena.hpp"
#include "BitMth/ia/Activations.hpp"
#include "BitMth/ia/LossFunctions.hpp"
#include "BitMth/ia/types/ActivationTypes.hpp"
#include "nn/Layer/types/LayerTypes.hpp"
#include <BitMth/io/DataNpyIO.hpp>
#include <BitMth/ia/IA.hpp>
#include <BitMth/linalg/Matrix.hpp>

#include <cmath>
#include <cstdint>
#include <nn/NeuralNetwork.hpp>
#include <nn/Layer/DenseLayer.hpp>
int main()
{
    using BitMth::linalg::Matrix;
    BitMth::io::DataNpyIO<uint8_t> mnistReader;
    std::mt19937 gen;
    Matrix<float> imagesTrain;
    Matrix<float> labelsTrain;
    Matrix<float> imagesTest;
    Matrix<float> labelsTest;
    BitMth::core::Arena arena(BitMth::core::Arena::MB(250));
    BitMth::core::Arena scratchArena(BitMth::core::Arena::MB(160));

    // 1. CARGA DEL DATASET DE ENTRENAMIENTO (60,000 imágenes)
    std::cout << "Cargando dataset de entrenamiento en matrices de BitMth..." << std::endl;
    {
        Matrix<uint8_t> imgTrainRead = mnistReader.read("datasets/mnist_X_train.npy");
        Matrix<uint8_t> labelsTrainRead = mnistReader.read("datasets/mnist_y_train.npy");
        Matrix<uint8_t> imgTestRead = mnistReader.read("datasets/mnist_X_test.npy");
        Matrix<uint8_t> labelsTestRead = mnistReader.read("datasets/mnist_y_test.npy");

        imagesTrain = Matrix<float>(imgTrainRead.getRows(),imgTrainRead.getCols(),&arena,true);
        labelsTrain = Matrix<float>(labelsTrainRead.getRows(),10,&arena,true);
        imagesTest = Matrix<float>(imgTestRead .getRows(),imgTestRead .getCols(),&arena,true);
        labelsTest = Matrix<float>(labelsTestRead .getRows(),10,&arena,true);

        constexpr float inv255 = 1.0f / 255.0f;

        const uint8_t* srcTrain = imgTrainRead.getValues();
        float* destTrain = imagesTrain.getValues();
        size_t totalTrainPixels = imagesTrain.size();
        for (size_t i = 0; i < totalTrainPixels; ++i) {
            destTrain[i] = static_cast<float>(srcTrain[i]) * inv255;
        }

        const uint8_t* srcTest = imgTestRead.getValues();
        float* destTest = imagesTest.getValues();
        size_t totalTestPixels = imagesTest.size();
        for (size_t i = 0; i < totalTestPixels; ++i) {
            destTest[i] = static_cast<float>(srcTest[i]) * inv255;
        }

        const uint8_t* rawTrainLabels = labelsTrainRead.getValues();
        for (size_t i = 0; i < labelsTrain.getRows(); ++i) {
            uint8_t labelClass = rawTrainLabels[i];
            labelsTrain(i, labelClass) = 1.0f;
        }

        const uint8_t* rawTestLabels = labelsTestRead.getValues();
        for (size_t i = 0; i < labelsTest.getRows(); ++i) {
            uint8_t labelClass = rawTestLabels[i];
            labelsTest(i, labelClass) = 1.0f;
        }

    }

    // 2. CONFIGURACIÓN DE LA RED NEURONAL
    NN::NeuralNetwork<float> nn(BitMth::ia::types::LossFunctType::MSE,BitMth::ia::types::OptimizerType::ADAM);
    nn.setLayers({
        {imagesTrain.getCols(), 128, NN::layer::types::LayerT::DENSE, BitMth::ia::types::ActivationFunctType::RELU},
        {128, 64, NN::layer::types::LayerT::DENSE, BitMth::ia::types::ActivationFunctType::RELU},
        {64, 10, NN::layer::types::LayerT::DENSE, BitMth::ia::types::ActivationFunctType::SIGMOID}
    });
                        
    // 3. BUCLE DE ENTRENAMIENTO EN MINI-BATCHES (Estabilidad Numérica)
    int epochs = 5;
    int batchSize = 64; 
    float learningRate = 0.001f;

    int totalBatches = std::ceil(float(imagesTrain.getRows()) / batchSize);
    std::cout << "Iniciando entrenamiento..." << std::endl;

    BitMth::ia::DataLoader<float> trainData(imagesTrain,labelsTrain,batchSize);
    BitMth::ia::DataLoader<float> testData(imagesTest,labelsTest,1);

    for (int epoch = 0; epoch < epochs; epoch++) {
        BitMth::linalg::Matrix<float> epochLoss(1,1) ;
        int numBatches = 0;

        while (trainData.hasNext()) {
            std::pair<Matrix<float>,Matrix<float>> batch = trainData.getNextBatch();

            nn.train(learningRate, batch.first, batch.second, &scratchArena);

            Matrix<float> prediction = nn.forwardPass(batch.first);
            
            epochLoss += BitMth::ia::LossFunctions<float>::mse(prediction, batch.second);
            numBatches++;
            printf("Epoch: %d, Batch: %d/%d, currentLoss: %f \r", epoch, totalBatches,numBatches, (epochLoss(0,0) / numBatches));
            std::fflush(stdout); 
            scratchArena.reset();
        }
        std::cout << std::endl;

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                  << " | Avg Loss: " << (epochLoss / numBatches) << "\r" << std::endl;
        trainData.reset(gen);
    }

    // 4. VALIDACIÓN CON EL DATASET DE TEST (10,000 imágenes para evaluar el Accuracy real)
    std::cout << "\nCargando dataset de validación (Test)..." << std::endl;

    int corrects = 0;

    while (testData.hasNext()) {

        std::pair<Matrix<float>,Matrix<float>> batch = testData.getNextBatch();
        Matrix<float> result = nn.forwardPass(batch.first);

        int highestIndex = 0;
        float maxVal = result(0, 0);
        for (int j = 1; j < 10; j++) {
            if (result(0, j) > maxVal) {
                maxVal = result(0, j);
                highestIndex = j;
            }
        }
        int actualLabel = 0;
        float maxLabelVal = batch.second(0, 0);
        for (int j = 1; j < 10; j++) {
            if (batch.second(0, j) > maxLabelVal) {
                maxLabelVal = batch.second(0, j);
                actualLabel = j;
            }
        }

        if (highestIndex == actualLabel) {
            corrects++;
        }
    }

    std::cout << "\n--- RESULTADO DE LA EVALUACIÓN ---" << std::endl;
    std::cout << "Muestras correctas: " << corrects << " / " << imagesTest.getRows() << std::endl;
    std::cout << "Precisión Final (Accuracy): " << (static_cast<float>(corrects) / imagesTest.getRows()) * 100.0f << "%" << std::endl;

    return 0;
}
