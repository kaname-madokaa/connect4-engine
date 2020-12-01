#pragma once
#include "Layer.h"
#include <iostream>


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
		for (auto x : m_filters)
		{
			for (auto y : x)
			{
				std::cout << y << "\n\n";
			}
		}

		for (auto x : m_biases)
		{
			std::cout << x << " ";
		}
		std::cout << "\n\n";
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

