#include "NN.h"
#include <fstream>
#include <iostream>

struct layerInfo
{
	//used when layer is dense
	int layerSize = -1;
	//used when layer is conv
	int filter_rows = -1;
	int filter_cols = -1;
	int input_rows = -1;
	int input_cols = -1;
	int input_channels = -1;
	int output_channels = -1;
};

void tanh_m(Eigen::MatrixXf& m)
{
	for (auto& val : m.reshaped())
	{
		val = std::tanh(val);
	}
}
void ReLU(Eigen::MatrixXf& m)
{
	for (auto& val : m.reshaped())
	{
		val = val > 0 ? val : 0;
	}
}

NN::NN(std::string dir)
{
	//weights file has the following structure:
	//first 4 bytes is the integer with the number of layers
	//then next number of layers bytes are layer types (one char/byte for each layer)
	//conv layers include the following information:
	//filter_rows, filter_cols, input_rows, input_cols, input_channels, output_channels (6*4 bytes in total)
	//dense layers include the following information:
	//iayer size
	//then the weights are put one after another in column major fashion
	//after that the biases

	std::ifstream weightsFile(dir, std::ios::binary);
	int layerNum;
	weightsFile.read((char*)&layerNum, sizeof(int));

	char* layerTypes = new char[layerNum];
	weightsFile.read(layerTypes, layerNum);

	std::vector<layerInfo> layersInformation;

	for (int layer = 0; layer < layerNum; layer++)
	{
		layerInfo currLayerInfo;
		if (layerTypes[layer] == 'c')
		{
			int* layerInfoArray = new int[6];
			weightsFile.read((char*)layerInfoArray, 6 * sizeof(int));
			currLayerInfo.filter_rows = layerInfoArray[0];
			currLayerInfo.filter_cols = layerInfoArray[1];
			currLayerInfo.input_rows = layerInfoArray[2];
			currLayerInfo.input_cols = layerInfoArray[3];
			currLayerInfo.input_channels = layerInfoArray[4];
			currLayerInfo.output_channels = layerInfoArray[5];
			currLayerInfo.layerSize = currLayerInfo.output_channels * (currLayerInfo.input_rows - currLayerInfo.filter_rows + 1) *
				(currLayerInfo.input_cols - currLayerInfo.filter_cols + 1);
			delete[] layerInfoArray;
		}
		else if (layerTypes[layer] == 'd')
		{
			int layerSize;
			weightsFile.read((char*)&layerSize, sizeof(int));
			currLayerInfo.layerSize = layerSize;
		}
		layersInformation.push_back(currLayerInfo);
	}

	for (int layer = 0; layer < layerNum; layer++)
	{
		if (layerTypes[layer] == 'c')
		{
			std::vector<std::vector<Eigen::MatrixXf>> weights;
			weights.resize(layersInformation[layer].input_channels);
			for (auto& channel : weights)
			{
				channel.resize(layersInformation[layer].output_channels);
				for (auto& w : channel)
				{
					w.resize(layersInformation[layer].filter_rows, layersInformation[layer].filter_cols);
				}
			}
			int currLayerWeightsSize = layersInformation[layer].filter_rows * layersInformation[layer].filter_cols *
				layersInformation[layer].input_channels * layersInformation[layer].output_channels;
			float* weightsData = new float[currLayerWeightsSize];
			weightsFile.read((char*)weightsData, currLayerWeightsSize * sizeof(float));

			int index = 0;
			for (int filter_row = 0; filter_row < layersInformation[layer].filter_rows; filter_row++)
			{
				for (int filter_col = 0; filter_col < layersInformation[layer].filter_cols; filter_col++)
				{
					for (int input_channel = 0; input_channel < layersInformation[layer].input_channels; input_channel++)
					{
						for (int output_channel = 0; output_channel < layersInformation[layer].output_channels; output_channel++)
						{
							weights[input_channel][output_channel](filter_row, filter_col) = weightsData[index++];
						}
					}
				}
			}
			std::vector<float> biases;
			biases.resize(layersInformation[layer].output_channels);
			weightsFile.read((char*)biases.data(), layersInformation[layer].output_channels * sizeof(float));
			std::unique_ptr<ConvLayer> currLayer = std::make_unique<ConvLayer>(weights, biases, layersInformation[layer].input_rows, layersInformation[layer].input_cols);
			m_layers.push_back(std::move(currLayer));
			delete[] weightsData;
		}
		else if (layerTypes[layer] == 'd')
		{
			Eigen::MatrixXf weights;
			weights.resize(layersInformation[layer].layerSize, layersInformation[layer - 1].layerSize);
			float* weightsData = new float[weights.size()];
			weightsFile.read((char*)weightsData, weights.size() * sizeof(float));
			for (int weight = 0; weight < weights.size(); weight++)
			{
				float z = weightsData[weight];
				weights.data()[weight] = weightsData[weight];
			}
			Eigen::MatrixXf biases;
			float* biasData = new float[layersInformation[layer].layerSize];
			biases.resize(layersInformation[layer].layerSize, 1);
			weightsFile.read((char*)biasData, layersInformation[layer].layerSize * sizeof(float));
			for (int weight = 0; weight < layersInformation[layer].layerSize; weight++)
			{
				biases.data()[weight] = biasData[weight];
			}

			if (layersInformation[layer].layerSize != 1)
			{
				std::unique_ptr<DenseLayer> currLayer = std::make_unique<DenseLayer>(weights, biases, ReLU, layersInformation[layer - 1].output_channels);
				m_layers.push_back(std::move(currLayer));
			}
			else
			{
				std::unique_ptr<DenseLayer> currLayer = std::make_unique<DenseLayer>(weights, biases, tanh_m, layersInformation[layer - 1].output_channels);
				m_layers.push_back(std::move(currLayer));
			}
			delete[] biasData;
			delete[] weightsData;
		}
	}
	weightsFile.close();
	for (int i = 0; i < layerNum; i++)
	{
		//m_layers[i]->printWeights();
	}
	//std::cout << "\n\n";
	delete[] layerTypes;

	lastRecordedPly = 0;
}

NN::NN(NN&& other) noexcept
{
	*this = std::move(other);
}

NN& NN::operator=(NN&& other) noexcept
{
	for (int layer = 0; layer < m_layers.size(); layer++)
	{
		m_layers[layer] = std::move(other.m_layers[layer]);
	}
	return *this;
}


Eigen::MatrixXf& NN::calculateOutput(const Board& pos)
{
	Eigen::MatrixXf input = boardToInput(pos);
	const Eigen::MatrixXf* previousLayerOutput;
	previousLayerOutput = &(m_layers.at(0)->forward(input));
	//std::cout << *previousLayerOutput << "\n\n";


	for (int Layer = 1; Layer < m_layers.size(); Layer++)
	{

		previousLayerOutput = &m_layers[Layer]->forward(*previousLayerOutput);
		//std::cout << *previousLayerOutput << "\n\n";

	}
#ifndef selfplay
	//std::cout << *previousLayerOutput << "\n\n";
#endif
	return *((Eigen::MatrixXf*)previousLayerOutput);
}

Eigen::MatrixXf& NN::calculateOutput(const Eigen::MatrixXf& input)
{
	const Eigen::MatrixXf* previousLayerOutput;
	previousLayerOutput = &(m_layers.at(0)->forward(input));
	//std::cout << *previousLayerOutput << "\n\n";


	for (int Layer = 1; Layer < m_layers.size(); Layer++)
	{

		previousLayerOutput = &m_layers[Layer]->forward(*previousLayerOutput);
		//std::cout << *previousLayerOutput << "\n\n";

	}
	//std::cout << *previousLayerOutput << "\n\n";
	return *((Eigen::MatrixXf*)previousLayerOutput);
}


const Eigen::MatrixXf& NN::boardToInput(const Board& pos)
{
	/*
	for (int row = 0; row < Board::rows; row++)
	{
		for (int col = 0; col < Board::cols * 3; col++)
		{
			for (int inputNum = 0; inputNum < pos.inputHistory.size(); inputNum++)
			{
				input(row, col + inputNum * Board::cols * 3) = pos.inputHistory[inputNum](row, col);
			}

			input(row, col) = (pos[COLOR_YELLOW] & (1ULL << row * Board::cols + col)) != 0;
			input(row, col + Board::cols) = (pos[COLOR_RED] & (1ULL << row * Board::cols + col)) != 0;
			input(row, col + Board::cols * 2) = (((pos[COLOR_RED] | pos[COLOR_YELLOW]) & (1ULL << row * Board::cols + col)) != 0);

			input(row, col + Board::cols * 3) = (pos[COLOR_YELLOW] & (1ULL << row * Board::cols + col)) == 0;
			input(row, col + Board::cols * 4) = (pos[COLOR_RED] & (1ULL << row * Board::cols + col)) == 0;
			input(row, col + Board::cols * 5) = (((pos[COLOR_RED] | pos[COLOR_YELLOW]) & (1ULL << row * Board::cols + col)) == 0);

			input(row, col + Board::cols * 6) = (pos[COLOR_YELLOW] & (1ULL << row * Board::cols + col)) != 0;
			input(row, col + Board::cols * 7) = (pos[COLOR_RED] & (1ULL << row * Board::cols + col)) != 0;
			input(row, col + Board::cols * 8) = (((pos[COLOR_RED] | pos[COLOR_YELLOW]) & (1ULL << row * Board::cols + col)) == 0);

			input(row, col + Board::cols * 9) = (pos[COLOR_YELLOW] & (1ULL << row * Board::cols + col)) == 0;
			input(row, col + Board::cols * 10) = (pos[COLOR_RED] & (1ULL << row * Board::cols + col)) == 0;
			input(row, col + Board::cols * 11) = (((pos[COLOR_RED] | pos[COLOR_YELLOW]) & (1ULL << row * Board::cols + col)) != 0);

		}
	}
	*/
	//std::cout << pos.get_NN_input().transpose() << "\n\n";
#if EVAL_NN || POLICY_NN
	return pos.get_NN_input();
#else
	return Eigen::MatrixXf();
#endif
}

