#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include "../Matrix/Matrix.hpp"

class ReadDataSet
{
public:
  int magicNumber = 0;
  int samples = 0;
  int width = 0;
  int height = 0;
  bool isImage;

  ReadDataSet(std::string path, bool isImage) : reader(path, std::ios::binary), isImage(isImage)
  {
    if (this->isImage)
    {
      headerSize = 16;
      reader.read(reinterpret_cast<char *>(&bytes), 4);
      magicNumber = int4(bytes);
      reader.read(reinterpret_cast<char *>(&bytes), 4);
      samples = int4(bytes);
      reader.read(reinterpret_cast<char *>(&bytes), 4);
      width = int4(bytes);
      reader.read(reinterpret_cast<char *>(&bytes), 4);
      height = int4(bytes);
    }
    else
    {
      headerSize = 8;
      reader.read(reinterpret_cast<char *>(&bytes), 4);
      magicNumber = int4(bytes);
      reader.read(reinterpret_cast<char *>(&bytes), 4);
      samples = int4(bytes);
    }
  };
  ~ReadDataSet() = default;

  void reset()
  {
    reader.clear();
    reader.seekg(headerSize, std::ios::beg);
  }
  unsigned char *readNextImage()
  {
    if (!this->isImage)
      return nullptr;

    if (!reader.read(reinterpret_cast<char *>(image), 784))
      return nullptr;

    return image;
  }

  unsigned char readNextLabel(unsigned char &outLabel)
  {
    if (isImage)
      return false;

    if (!reader.read(reinterpret_cast<char *>(&label), 1))
      return false;

    outLabel = label;
    return true;
  }

  Matrix<float> getNextBatchImages(int bacthSize)
  {
    Matrix<float> X(784, bacthSize);

    for (size_t i = 0; i < bacthSize; i++)
    {
      unsigned char *image = this->readNextImage();
      if (!image)
      {
        // Si se acabó el archivo, reiniciamos y seguimos
        reset();
        image = readNextImage();
      }
      for (size_t j = 0; j < 784; j++)
      {
        X[j][i] = image[j] / 255.0f;
      }
    }
    return X;
  }

  Matrix<float> getNextBatchLabes(int bacthSize)
  {
    Matrix<float> Y(10, bacthSize);
    for (size_t i = 0; i < bacthSize; i++)
    {
      unsigned char label;
      if (!readNextLabel(label))
      {
        reset();
        readNextLabel(label);
      }
      for (size_t j = 0; j < 10; j++)
      {
        if (label == j)
        {
          Y[j][i] = 1.0f;
        }
        else
        {
          Y[j][i] = 0;
        }
      }
    }
    return Y;
  }

private:
  unsigned char bytes[4];
  std::ifstream reader;
  unsigned char image[784];
  unsigned char label;
  int headerSize = 0;

  int int4(unsigned char bytes[4])
  {
    int value = 0;
    value = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
    return value;
  }
};
