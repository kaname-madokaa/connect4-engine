#pragma once
#include "Layer.h"
#include <iostream>
class DenseLayer :
    public Layer
{
public:
    DenseLayer(const Eigen::MatrixXf& weights, const Eigen::MatrixXf& biases,  std::function<void(Eigen::MatrixXf&)> activation_function =
        [&](Eigen::MatrixXf& matrix) {for (auto& val : matrix.reshaped())
    {
        val = val > 0 ? val : 0;
    }}, int input_layer_channels = 20) : m_activation_function(activation_function), m_input_layer_channels(input_layer_channels)
    {
        m_weights = weights;
        m_biases = biases;
    }
    const Eigen::MatrixXf& forward(const Eigen::MatrixXf& input)
    {
        Eigen::MatrixXf newMatrix;

        if (input.cols() > 1)
        {
            newMatrix.resize(input.size(), 1);
            int index = 0;
            for (int row = 0; row < input.rows(); row++)
            {
                for (int col = 0; col < input.cols() / m_input_layer_channels; col++)
                {
                    for (int channel = 0; channel < m_input_layer_channels; channel++)
                    {
                        newMatrix(index++, 0) = input(row, col + channel * input.cols() / m_input_layer_channels);
                    }
                }
            }
        }
        else
        {
            newMatrix = input;
        }
        m_output = m_weights * newMatrix + m_biases;
        m_activation_function(m_output);
        return m_output;
    }
    void printWeights()
    {
        std::cout << m_weights << "\n\n";
        std::cout << m_biases << "\n\n";
    }
    char layerType()
    {
        return 'd';
    }
private:
    Eigen::MatrixXf m_output;
    Eigen::MatrixXf m_weights;
    Eigen::MatrixXf m_biases;
    std::function<void(Eigen::MatrixXf&)> m_activation_function;

    int m_input_layer_channels;
    
};

