#ifndef EMBEDED
#include "utils.h"
#endif
#include "NN.h"
#include <fstream>
#include <iostream>



#ifdef EMBEDED
void tanh_m(Eigen::MatrixXf& matrix)
{
	for (int row = 0; row < matrix.rows(); row++)
	{
		for (int col = 0; col < matrix.cols(); col++)
		{
			matrix(row, col) = std::tanh(matrix(row, col));
		}
	}
}


void ReLU(Eigen::MatrixXf& matrix)
{
	for (int row = 0; row < matrix.rows(); row++)
	{
		for (int col = 0; col < matrix.cols(); col++)
		{
			matrix(row, col) = matrix(row, col) > 0 ? matrix(row, col) : 0;
		}
	}
}
#endif

NN::NN(std::string dir)
{
#ifdef EMBEDED
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
			currLayerInfo.layerSize = currLayerInfo.output_channels * (currLayerInfo.input_rows - currLayerInfo.filter_rows +1) * 
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
				std::unique_ptr<DenseLayer> currLayer = std::make_unique<DenseLayer>(weights, biases,ReLU, layersInformation[layer-1].output_channels);
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
	
#else
/*
weights.resize(layersInformation[layer].input_channels);
for (auto& channel : weights)
{
	channel.resize(layersInformation[layer].output_channels);
	for (auto& w : channel)
	{
		w.resize(layersInformation[layer].filter_rows, layersInformation[layer].filter_cols);
	}
}*/
//layer 1
//weights file has the following structure:
	//first 4 bytes is the integer with the number of layers
	//then next number of layers bytes are layer types (one char/byte for each layer)
	//conv layers include the following information:
	//filter_rows, filter_cols, input_rows, input_cols, input_channels, output_channels (6*4 bytes in total)
	//dense layers include the following information:
	//iayer size
	//then the weights are put one after another in column major fashion
	//after that the biases

	std::ifstream weightsFile("C:/Users/Anastazja/abcd", std::ios::binary);
	int ab;
	weightsFile.read((char*)&ab, sizeof(int));

	char layerTypes[100];
	weightsFile.read(layerTypes, layerNum);
	char a[1000];
	weightsFile.read(a, 6 * 6 * 4 + 2 * 4);

	Eigen::Matrix<float, layersInformation[0].filter_rows, layersInformation[0].filter_cols> layer_0_weights_data[layersInformation[0].input_channels]
		[layersInformation[0].output_channels];
	unsigned long long index = 0;
	constexpr int currLayerWeightsSize = layersInformation[0].filter_rows * layersInformation[0].filter_cols *
		layersInformation[0].input_channels * layersInformation[0].output_channels;
	float layer_0_weights_data_float[currLayerWeightsSize];
	weightsFile.read((char*)layer_0_weights_data_float, currLayerWeightsSize * sizeof(float));

	for (int filter_row = 0; filter_row < layersInformation[0].filter_rows; filter_row++)
	{
		for (int filter_col = 0; filter_col < layersInformation[0].filter_cols; filter_col++)
		{
			for (int input_channel = 0; input_channel < layersInformation[0].input_channels; input_channel++)
			{
				for (int output_channel = 0; output_channel < layersInformation[0].output_channels; output_channel++)
				{
					layer_0_weights_data[input_channel][output_channel](filter_row, filter_col) = layer_0_weights_data_float[index++];
				}
			}
		}
	}
	float layer_0_biases_data[layersInformation[0].output_channels];
	weightsFile.read((char*)layer_0_biases_data, layersInformation[0].output_channels * sizeof(float));
	firstLayer = CONV_LAYER(0)(layer_0_weights_data, layer_0_biases_data);

//layer 2
	Eigen::Matrix<float, layersInformation[1].filter_rows, layersInformation[1].filter_cols> layer_1_weights_data[layersInformation[1].input_channels]
		[layersInformation[1].output_channels];
	index = 0;
	constexpr int currLayerWeightsSize1 = layersInformation[1].filter_rows * layersInformation[1].filter_cols *
		layersInformation[1].input_channels * layersInformation[1].output_channels;
	float layer_1_weights_data_float[currLayerWeightsSize1];
	weightsFile.read((char*)layer_1_weights_data_float, currLayerWeightsSize1 * sizeof(float));
	for (int filter_row = 0; filter_row < layersInformation[1].filter_rows; filter_row++)
	{
		for (int filter_col = 0; filter_col < layersInformation[1].filter_cols; filter_col++)
		{
			for (int input_channel = 0; input_channel < layersInformation[1].input_channels; input_channel++)
			{
				for (int output_channel = 0; output_channel < layersInformation[1].output_channels; output_channel++)
				{
					layer_1_weights_data[input_channel][output_channel](filter_row, filter_col) = layer_1_weights_data_float[index++];
				}
			}
		}
	}
	float layer_1_biases_data[layersInformation[1].output_channels];
	weightsFile.read((char*)layer_1_biases_data, layersInformation[1].output_channels * sizeof(float));
	secondLayer = CONV_LAYER(1)(layer_1_weights_data, layer_1_biases_data);

//layer 3
	Eigen::Matrix<float, layersInformation[2].filter_rows, layersInformation[2].filter_cols> layer_2_weights_data[layersInformation[2].input_channels]
		[layersInformation[2].output_channels];
	index = 0;
	constexpr int currLayerWeightsSize2 = layersInformation[2].filter_rows * layersInformation[2].filter_cols *
		layersInformation[2].input_channels * layersInformation[2].output_channels;
	float layer_2_weights_data_float[currLayerWeightsSize2];
	weightsFile.read((char*)layer_2_weights_data_float, currLayerWeightsSize2 * sizeof(float));
	for (int filter_row = 0; filter_row < layersInformation[2].filter_rows; filter_row++)
	{
		for (int filter_col = 0; filter_col < layersInformation[2].filter_cols; filter_col++)
		{
			for (int input_channel = 0; input_channel < layersInformation[2].input_channels; input_channel++)
			{
				for (int output_channel = 0; output_channel < layersInformation[2].output_channels; output_channel++)
				{
					layer_2_weights_data[input_channel][output_channel](filter_row, filter_col) = layer_2_weights_data_float[index++];
				}
			}
		}
	}
	float layer_2_biases_data[layersInformation[2].output_channels];
	weightsFile.read((char*)layer_2_biases_data, layersInformation[2].output_channels * sizeof(float));
	thirdLayer = CONV_LAYER(2)(layer_2_weights_data, layer_2_biases_data);


//layer 4
	Eigen::Matrix<float, layersInformation[3].filter_rows, layersInformation[3].filter_cols> layer_3_weights_data[layersInformation[3].input_channels]
		[layersInformation[3].output_channels];
	index = 0;
	constexpr int currLayerWeightsSize3 = layersInformation[3].filter_rows * layersInformation[3].filter_cols *
		layersInformation[3].input_channels * layersInformation[3].output_channels;
	float layer_3_weights_data_float[currLayerWeightsSize3];
	weightsFile.read((char*)layer_3_weights_data_float, currLayerWeightsSize3 * sizeof(float));
	for (int filter_row = 0; filter_row < layersInformation[3].filter_rows; filter_row++)
	{
		for (int filter_col = 0; filter_col < layersInformation[3].filter_cols; filter_col++)
		{
			for (int input_channel = 0; input_channel < layersInformation[3].input_channels; input_channel++)
			{
				for (int output_channel = 0; output_channel < layersInformation[3].output_channels; output_channel++)
				{
					layer_3_weights_data[input_channel][output_channel](filter_row, filter_col) = layer_3_weights_data_float[index++];
				}
			}
		}
	}
	float layer_3_biases_data[layersInformation[3].output_channels];
	weightsFile.read((char*)layer_3_biases_data, layersInformation[3].output_channels * sizeof(float));
	fourthLayer = CONV_LAYER(3)(layer_3_weights_data, layer_3_biases_data);
	
//layer 5
	/*
Eigen::MatrixXf weights;
weights.resize(layersInformation[layer].layerSize, layersInformation[layer - 1].layerSize);
Eigen::MatrixXf weights;
weights.resize(layersInformation[layer].layerSize, layersInformation[layer - 1].layerSize);
float* weightsData = new float[weights.size()];
weightsFile.read((char*)weightsData, weights.size() * sizeof(float));
for (int weight = 0; weight < weights.size(); weight++)
{
	float z = weightsData[weight];
	weights.data()[weight] = weightsData[weight];
}
}*/
	Eigen::Matrix<float, layersInformation[4].filter_rows, layersInformation[4].filter_cols> layer_4_weights_data[layersInformation[4].input_channels]
		[layersInformation[4].output_channels];
	index = 0;
	constexpr int currLayerWeightsSize4 = layersInformation[4].filter_rows * layersInformation[4].filter_cols *
		layersInformation[4].input_channels * layersInformation[4].output_channels;
	float layer_4_weights_data_float[currLayerWeightsSize4];
	weightsFile.read((char*)layer_4_weights_data_float, currLayerWeightsSize4 * sizeof(float));
	for (int filter_row = 0; filter_row < layersInformation[4].filter_rows; filter_row++)
	{
		for (int filter_col = 0; filter_col < layersInformation[4].filter_cols; filter_col++)
		{
			for (int input_channel = 0; input_channel < layersInformation[4].input_channels; input_channel++)
			{
				for (int output_channel = 0; output_channel < layersInformation[4].output_channels; output_channel++)
				{
					layer_4_weights_data[input_channel][output_channel](filter_row, filter_col) = layer_4_weights_data_float[index++];
				}
			}
		}
	}
	float layer_4_biases_data[layersInformation[4].output_channels];
	weightsFile.read((char*)layer_4_biases_data, layersInformation[4].output_channels * sizeof(float));
	fifthLayer = CONV_LAYER(4)(layer_4_weights_data, layer_4_biases_data);
//layer 6
	Eigen::Matrix<float, layersInformation[5].filter_rows, layersInformation[5].filter_cols> layer_5_weights_data[layersInformation[5].input_channels]
		[layersInformation[5].output_channels];
	index = 0;
	constexpr int currLayerWeightsSize5 = layersInformation[5].filter_rows * layersInformation[5].filter_cols *
		layersInformation[5].input_channels * layersInformation[5].output_channels;
	float layer_5_weights_data_float[currLayerWeightsSize5];
	weightsFile.read((char*)layer_5_weights_data_float, currLayerWeightsSize5 * sizeof(float));
	for (int filter_row = 0; filter_row < layersInformation[5].filter_rows; filter_row++)
	{
		for (int filter_col = 0; filter_col < layersInformation[5].filter_cols; filter_col++)
		{
			for (int input_channel = 0; input_channel < layersInformation[5].input_channels; input_channel++)
			{
				for (int output_channel = 0; output_channel < layersInformation[5].output_channels; output_channel++)
				{
					layer_5_weights_data[input_channel][output_channel](filter_row, filter_col) = layer_5_weights_data_float[index++];
				}
			}
		}
	}
	float layer_5_biases_data[layersInformation[5].output_channels];
	weightsFile.read((char*)layer_5_biases_data, layersInformation[5].output_channels * sizeof(float));
	sixthLayer = CONV_LAYER_1x1(5)(layer_5_weights_data, layer_5_biases_data);

//layer 7
	Eigen::Matrix<float, layersInformation[6].layerSize, layersInformation[6 - 1].layerSize> layer_6_weights_data;
	index = 0;
	float layer_6_weights_data_raw[layersInformation[6].layerSize * layersInformation[6 - 1].layerSize];
	weightsFile.read((char*)layer_6_weights_data_raw, layersInformation[6].layerSize * layersInformation[6 - 1].layerSize * sizeof(float));
	for (int weight = 0; weight < layer_6_weights_data.size(); weight++)
	{
		layer_6_weights_data.data()[weight] = layer_6_weights_data_raw[weight];
	}
	Eigen::Matrix<float, layersInformation[6].layerSize, 1> layer_6_biases_data;
	float layer_6_biases_data_raw[layersInformation[6].layerSize];
	weightsFile.read((char*)layer_6_biases_data_raw, layersInformation[6].layerSize * sizeof(float));
	for (int weight = 0; weight < layersInformation[6].layerSize; weight++)
	{
		layer_6_biases_data.data()[weight] = layer_6_biases_data_raw[weight];
	}
	firstDenseLayer = DENSE_LAYER(6)(layer_6_weights_data, layer_6_biases_data);

//layer 8
	Eigen::Matrix<float, layersInformation[7].layerSize, layersInformation[7 - 1].layerSize> layer_7_weights_data;
	index = 0;
	float layer_7_weights_data_raw[layersInformation[7].layerSize * layersInformation[7 - 1].layerSize];
	weightsFile.read((char*)layer_7_weights_data_raw, layersInformation[7].layerSize * layersInformation[7 - 1].layerSize * sizeof(float));
	for (int weight = 0; weight < layer_7_weights_data.size(); weight++)
	{
		layer_7_weights_data.data()[weight] = layer_7_weights_data_raw[weight];
	}
	Eigen::Matrix<float, layersInformation[7].layerSize, 1> layer_7_biases_data;
	float layer_7_biases_data_raw[layersInformation[7].layerSize];
	weightsFile.read((char*)layer_7_biases_data_raw, layersInformation[7].layerSize * sizeof(float));
	for (int weight = 0; weight < layersInformation[7].layerSize; weight++)
	{
		std::cout << layer_7_biases_data_raw[weight] << "\n";
		layer_7_biases_data.data()[weight] = layer_7_biases_data_raw[weight];
	}
	secondDenseLayer = DENSE_LAYER(7)(layer_7_weights_data, layer_7_biases_data,
			[&](Eigen::Matrix<float, layersInformation[7].layerSize, 1>& matrix) {for (auto& val : matrix.reshaped())
	{
		val = std::tanh(val);
	}});


#endif

	lastRecordedPly = 0;
}

#ifdef EMBEDED

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
#else
const Eigen::Matrix<float, 1, 1>& NN::calculateOutput(const Board& pos)
{
	return calculateOutput(pos.get_NN_input());
}
const Eigen::Matrix<float, 1, 1>& NN::calculateOutput(const Eigen::Matrix<float, input_rows, input_cols * input_channels>& input)
{
	const auto& firstLayerOutput = firstLayer.forward(input);
	const auto& secondLayerOutput = secondLayer.forward(firstLayerOutput);
	const auto& thirdLayerOutput = thirdLayer.forward(secondLayerOutput);
	const auto& fourthLayerOutput = fourthLayer.forward(thirdLayerOutput);
	const auto& fifthLayerOutput = fifthLayer.forward(fourthLayerOutput);
	const auto& sixthLayerOutput = sixthLayer.forward(fifthLayerOutput);
	const auto& sixthLayerFlattened = sixthLayer.flatten();
	const auto& firstDenseLayerOutput = firstDenseLayer.forward(sixthLayerFlattened);
	const auto& secondDenseLayerOutput = secondDenseLayer.forward(firstDenseLayerOutput);
	return secondDenseLayerOutput;
}
const Eigen::Matrix<float, 6, 7 * input_channels>& NN::boardToInput(const Board& pos)
{
	return pos.get_NN_input();
}

#endif