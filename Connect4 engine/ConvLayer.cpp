#include "ConvLayer.h"
#include <fstream>
#ifdef EMBEDED
ConvLayer::ConvLayer(std::vector<std::vector<Eigen::MatrixXf>>& weights_data, std::vector<float>& bias_data, int input_rows, int input_cols, std::function<void(Eigen::MatrixXf&)> activation_function)
	: m_activation_function(activation_function)
{
	m_filters = std::move(weights_data);
	m_biases = std::move(bias_data);
	m_input_rows = input_rows;
	m_input_cols = input_cols;
	m_filter_rows = m_filters[0][0].rows();
	m_filter_cols = m_filters[0][0].cols();
	m_output_cols = (m_input_cols - m_filter_cols) + 1;
	m_output_rows = (m_input_rows - m_filter_rows) + 1;
	m_input_channels = m_filters.size();
}


const Eigen::MatrixXf& ConvLayer::forward(const Eigen::MatrixXf& input)
{
	//m_filters.size() - input channels
	//m_filters[0].size() - own filters
	m_output.resize(m_output_rows, m_output_cols * m_filters[0].size());
	m_output.setZero();
	for (int input_channel = 0; input_channel < m_input_channels; input_channel++)
	{
		for (int output_channel = 0; output_channel < m_filters[0].size(); output_channel++)
		{
			for (int row = 0; row < m_output_rows; row++)
			{
				for (int col = 0; col < m_output_cols; col++)
				{
					m_output(
						//row
						row,
						//col
						col + m_output_cols * output_channel
					) 
						+= m_filters[input_channel][output_channel].cwiseProduct(
						input.block(row, ((m_input_cols * input_channel) + col), m_filter_rows, m_filter_cols)).sum();
				}
			}
		}
	}
	for (int output_channel = 0; output_channel < m_filters[0].size(); output_channel++)
	{
		for (int row = 0; row < m_output_rows; row++)
		{
			for (int col = 0; col < m_output_cols; col++)
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
#endif