#pragma once
#include "Layer.h"
#include <iostream>
#include <cstring>
#include <fstream>

#ifdef EMBEDED
class ConvLayer: public Layer
{
public:
	const Eigen::MatrixXf& forward(const Eigen::MatrixXf& input);
	Eigen::MatrixXf& getOutput();
	ConvLayer(std::vector<std::vector<Eigen::MatrixXf>>& weights_data, std::vector<float>& bias_data, int input_rows, int input_cols, std::function<void(Eigen::MatrixXf&)> activation_function =
		[&](Eigen::MatrixXf& matrix) {for (auto& val : matrix.reshaped())
	{
		val = val > 0 ? val : 0;
	}});
	void printWeights()
	{
		std::ofstream filters("filters.txt", std::ios::app);
		for (auto x : m_filters)
		{
			for (auto y : x)
			{
				filters << y << "\n\n";
			}
		}

		for (auto x : m_biases)
		{
			filters << x << " ";
		}
		filters << "\n\n";
	}
	char layerType()
	{
		return 'c';
	}
private:
	std::vector<std::vector<Eigen::MatrixXf>> m_filters;
	std::vector<float> m_biases;
	std::function<void(Eigen::MatrixXf&)> m_activation_function = 
		[&](Eigen::MatrixXf& matrix) {for (auto& val : matrix.reshaped())
	{
		val = val > 0 ? val : 0;
	}
		};
	Eigen::MatrixXf m_output; 
	int m_input_rows;
	int m_input_cols;
	int m_input_channels;

	int m_filter_rows;
	int m_filter_cols;
	int m_output_cols;
	int m_output_rows;
};
#else
template <typename input_type, typename output_type, typename weights_matrix, typename bias_matrix, int input_rows, int input_cols,
	int input_channels, int filter_rows, int filter_cols, int output_cols, int output_rows, int output_channels, int layer_size >
	class ConvLayer
{
private:
	output_type m_output;
	weights_matrix m_filters[input_channels][output_channels];
	float m_biases[output_channels];
	std::function<void(output_type&)> m_activation_function;
	int m_input_rows;
	int m_input_cols;
	int m_input_channels;
	int m_layer_size;
	int m_filter_rows;
	int m_filter_cols;
	int m_output_cols;
	int m_output_rows;
	int m_output_channels;
	Eigen::Matrix<float, layer_size, 1> m_flattened_output;

public:
	output_type& getOutput()
	{
		return m_output;
	}
	Eigen::Matrix<float, layer_size, 1>& flatten()
	{
		unsigned int index = 0;
		for (int row = 0; row < output_rows; row++)
		{
			for (int col = 0; col < output_cols; col++)
			{
				for (int channel = 0; channel < output_channels; channel++)
				{
					m_flattened_output(index++, 0) = m_output(row, col + channel * output_cols);
 				}
			}
		}
		return m_flattened_output;
	}
	ConvLayer()
	{

	}
	ConvLayer(weights_matrix (&weights_data)[input_channels][output_channels], float (&bias_data)[output_channels], std::function<void(output_type&)> activation_function =
			[&](output_type& matrix) {for (auto& val : matrix.reshaped())
		{
			val = val > 0 ? val : 0;
		}
	}): m_input_rows(input_rows), m_input_cols(input_cols), m_input_channels(input_channels), m_filter_rows(filter_rows),
			m_filter_cols(filter_cols), m_output_cols(output_cols), m_output_rows(output_rows), m_output_channels(output_channels), m_activation_function(activation_function)
	{
		std::memcpy(m_filters, weights_data, input_channels * output_channels * sizeof(weights_matrix));
		std::memcpy(m_biases, bias_data, output_channels * sizeof(float));
	}

	const output_type& forward(const input_type& input)
	{
		m_output.setZero();
		for (int input_channel = 0; input_channel < input_channels; input_channel++)
		{
			for (int output_channel = 0; output_channel < output_channels; output_channel++)
			{
				for (int row = 0; row < output_rows; row++)
				{
					for (int col = 0; col < output_cols; col++)
					{
						m_output(
							//row
							row,
							//col
							col + output_cols * output_channel
						)
							+= m_filters[input_channel][output_channel].cwiseProduct(
								input.block<filter_rows, filter_cols>(row, (m_input_cols * input_channel) + col)).sum();
						
					}
				}
			}
		}
		for (int output_channel = 0; output_channel < output_channels; output_channel++)
		{
			for (int row = 0; row < output_rows; row++)
			{
				for (int col = 0; col < output_cols; col++)
				{
					m_output(
						//row
						row,
						//col
						col + m_output_cols * output_channel
					) += m_biases[output_channel];
				}
			}
		}
		m_activation_function(m_output);
		return m_output;
	}



};
#endif

