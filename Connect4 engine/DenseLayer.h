#pragma once
#include "Layer.h"
#include <iostream>
#include <fstream>


#ifdef EMBEDED
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
        printWeights();
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
        std::ofstream a("weights.txt", std::ios::app);
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
#else
template <typename input_type, typename output_type, typename weights_type, typename biases_type,  int input_layer_channels>
class DenseLayer 
{
public:
    DenseLayer(const weights_type& weights, const biases_type& biases, std::function<void(output_type&)> activation_function =
        [&](output_type& matrix) {for (auto& val : matrix.reshaped())
    {
        val = val > 0 ? val : 0;
    }}) : m_activation_function(activation_function), m_input_layer_channels(input_layer_channels)
    {
        m_weights = weights;
        m_biases = biases;
    }
    DenseLayer()
    {

    }

    const output_type& forward(const input_type& input)
    {
        m_output = m_weights * input + m_biases;
        m_activation_function(m_output);
        return m_output;
    }

private:
    output_type m_output;
    weights_type m_weights;
    biases_type m_biases;
    std::function<void(output_type&)> m_activation_function;

    int m_input_layer_channels;
};

#endif
