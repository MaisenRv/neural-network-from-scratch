#include "nn/NeuralNetwork.hpp"
#include "Matrix/Matrix.hpp"
#include <vector>
#include <iostream>
#include "utils/readDataSet.hpp"
// #include "BitMth/math/Activations.hpp"
int main()
{
  	ReadDataSet data("datasets/train-images-idx3-ubyte",true);
	ReadDataSet dataLabels("datasets/train-labels-idx1-ubyte",false);
	int dataLength = data.width * data.height;
	std::vector<int> shape = {dataLength, 128, 10};



	// Matrix<float> X(data.samples,dataLength);
	// for (size_t i = 0; i < data.samples; i++)
	// {
	// 	unsigned char* image = data.readNextImage();
	// 	for (size_t j = 0; j < dataLength; j++)
	// 	{
	// 		X[i][j] = image[j]/255.0f;	
	// 	}	
	// }

	
	// Matrix<float> Y(dataLabels.samples,10);
	// for (size_t i = 0; i < dataLabels.samples; i++)
	// {
	// 	unsigned char label = dataLabels.readNextLabel();
	// 	for (size_t j = 0; j < 10; j++)
	// 	{
	// 		if(label == j){
	// 			Y[i][j] = 1.0f;
	// 		}else{
	// 			Y[i][j] = 0;
	// 		}
				
	// 	}	
	// }
	
	int batchSize = 100;
	
	NeuralNetwork<float> NN(shape, "sigmoid", batchSize);
	// NN.show();
	// NN.train(0.1, X, Y);
	for (size_t i = 0; i < 5; i++)
	{
		data.reset();
    	dataLabels.reset();
		for (size_t i = 0; i < 600; i++)
		{
			NN.trainBatch(0.6,data.getNextBatchImages(batchSize),dataLabels.getNextBatchLabes(batchSize));
		}
		std::cout << "Epoch: " << i + 1 << std::endl;
	}
	
	
	
	




	ReadDataSet dataTest("datasets/t10k-images-idx3-ubyte",true);
	int dataLengthTest = dataTest.width * dataTest.height;
	Matrix<float> XTest(dataTest.samples,dataLengthTest);
	for (size_t i = 0; i < dataTest.samples; i++)
	{
		unsigned char* image = dataTest.readNextImage();
		for (size_t j = 0; j < dataLengthTest; j++)
		{
			XTest[i][j] = image[j]/255.0f;	
		}	
	}

	ReadDataSet dataLabelsTest("datasets/t10k-labels-idx1-ubyte",false);
	Matrix<float> YTest(dataLabelsTest.samples,10);
	for (size_t i = 0; i < dataLabelsTest.samples; i++)
	{
		unsigned char label; 
		dataLabelsTest.readNextLabel(label);
		for (size_t j = 0; j < 10; j++)
		{
			if(label == j){
				YTest[i][j] = 1.0f;
			}else{
				YTest[i][j] = 0;
			}
				
		}	
	}


	int corrects = 0;
	float totalLoss = 0.0f;
	int totalSamples = 0;

	batchSize = 100;

	for (int start = 0; start < dataTest.samples; start += batchSize)
	{
		int currentBatch = batchSize;
		if (start + batchSize > dataTest.samples)
			currentBatch = dataTest.samples - start;

		// Construyes XBatch (784, currentBatch) y YBatch (10, currentBatch)
		Matrix<float> XBatch(784, currentBatch);
		Matrix<float> YBatch(10, currentBatch);

		for (int i = 0; i < currentBatch; i++)
		{
			// Copiar sample i del dataset
			Matrix<float> x = XTest.getRow(start + i).t(); // (784,1)
			Matrix<float> y = YTest.getRow(start + i).t(); // (10,1)

			for (int r = 0; r < 784; r++)
				XBatch[r][i] = x[r][0];

			for (int r = 0; r < 10; r++)
				YBatch[r][i] = y[r][0];
		}

		Matrix<float> result = NN.forwardPass(XBatch); // (10, B)

		// Loss batch
		float loss = NeuralNetwork<float>::batchLossMSE(result, YBatch);
		totalLoss += loss * currentBatch;

		// Accuracy batch
		for (int c = 0; c < currentBatch; c++)
		{
			int p = Matrix<float>::argmaxCol(result, c);
			int y = Matrix<float>::argmaxCol(YBatch, c);

			if (p == y)
				corrects++;
		}

		totalSamples += currentBatch;

		if (start % 1000 == 0)
		{
			std::cout << "Acc: " << corrects << " / " << totalSamples
					<< " = " << ((float)corrects / (float)totalSamples)
					<< " | avgLoss: " << (totalLoss / (float)totalSamples)
					<< std::endl;
		}
	}

	std::cout << "\nFINAL TEST:\n";
	std::cout << "Accuracy: " << ((float)corrects / (float)totalSamples) << std::endl;
	std::cout << "Avg loss: " << (totalLoss / (float)totalSamples) << std::endl;






	// int corrects = 0;
	// float totalLoss = 0;

	// for (size_t i = 0; i < dataTest.samples; i++)
	// {
	// 	Matrix<float> x = XTest.getRow(i).t();
	// 	Matrix<float> y = YTest.getRow(i).t();

	// 	Matrix<float> result = NN.forwardPass(x);

	// 	// Loss
	// 	float loss = NN.errCuaMed(result, y);
	// 	totalLoss += loss;

	// 	// Accuracy (esto es lo importante)
	// 	int pred = Matrix<float>::argmax(result);
	// 	int real = Matrix<float>::argmax(y);

	// 	if (pred == real)
	// 		corrects++;

	// 	if (i % 100 == 0)
	// 	{
	// 		std::cout << "Acc: " << corrects << " / " << (i + 1)
	// 				<< " = " << ((float)corrects / (float)(i + 1))
	// 				<< " | avgLoss: " << (totalLoss / (float)(i + 1))
	// 				<< std::endl;
	// 	}
	// }

	// std::cout << "\nFINAL TEST:\n";
	// std::cout << "Accuracy: " << ((float)corrects / (float)dataTest.samples) << std::endl;
	// std::cout << "Avg loss: " << (totalLoss / (float)dataTest.samples) << std::endl;

	return 0;
}
