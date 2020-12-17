#pragma once
#include "Board.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "ConvLayer.h"
#include "DenseLayer.h"
#include "info.h"

#define CONV_LAYER(x) ConvLayer<Eigen::Matrix<float, layersInformation[x].input_rows, layersInformation[x].input_cols * layersInformation[x].input_channels>,Eigen::Matrix<float, layersInformation[x+1].input_rows, layersInformation[x+1].input_cols* layersInformation[x].output_channels>,Eigen::Matrix<float, layersInformation[x].filter_rows, layersInformation[x].filter_cols>,float,layersInformation[x].input_rows,layersInformation[x].input_cols,layersInformation[x].input_channels,layersInformation[x].filter_rows,layersInformation[x].filter_cols,layersInformation[x+1].input_cols,layersInformation[x+1].input_rows,layersInformation[x].output_channels,layersInformation[x].layerSize>
#define CONV_LAYER_1x1(x) ConvLayer<Eigen::Matrix<float, layersInformation[x].input_rows, layersInformation[x].input_cols* layersInformation[x].input_channels>,Eigen::Matrix<float, layersInformation[x].input_rows, layersInformation[x].input_cols* layersInformation[x].output_channels>,Eigen::Matrix<float, layersInformation[x].filter_rows, layersInformation[x].filter_cols>,float,layersInformation[x].input_rows,layersInformation[x].input_cols,layersInformation[x].input_channels,layersInformation[x].filter_rows,layersInformation[x].filter_cols,layersInformation[x].input_cols,layersInformation[x].input_rows,layersInformation[x].output_channels,layersInformation[x].layerSize>
#define DENSE_LAYER(x) DenseLayer<Eigen::Matrix<float, layersInformation[x - 1].layerSize, 1>,Eigen::Matrix<float, layersInformation[x].layerSize, 1>,Eigen::Matrix<float, layersInformation[x].layerSize, layersInformation[x - 1].layerSize>,Eigen::Matrix<float, layersInformation[x].layerSize, 1>,layersInformation[x - 1].output_channels>
constexpr int minibatch_size = 100;
constexpr double dropoutChance = 0.3;
constexpr double learningRate = 0.1;
constexpr float epsilon = 0.00000001f;

#ifdef EMBEDED
constexpr int input_channels = 3;
#endif

class NN
{
public:
	NN(std::string dir);
#ifdef EMBEDED
	NN() :NN("C:/Users/Anastazja/abcd") {}
	NN(NN&& other) noexcept;
	NN& operator=(NN&& other) noexcept;
	Eigen::MatrixXf& calculateOutput(const Board& pos);
	Eigen::MatrixXf& calculateOutput(const Eigen::MatrixXf& input);
	const Eigen::MatrixXf& boardToInput(const Board& pos);
#else
	const Eigen::Matrix<float,1,1>& calculateOutput(const Board& pos);
	const Eigen::Matrix<float, 1, 1>& calculateOutput(const Eigen::Matrix<float, input_rows, input_cols* input_channels>& input);
	const Eigen::Matrix<float, 6, 7 * input_channels>& boardToInput(const Board& pos);
#endif

private:
	int lastRecordedPly;
#ifdef EMBEDED
	std::vector<std::unique_ptr<Layer>> m_layers;
#else
	

	CONV_LAYER(0) firstLayer;
	CONV_LAYER(1) secondLayer;
	CONV_LAYER(2) thirdLayer;
	CONV_LAYER(3) fourthLayer;
	CONV_LAYER(4) fifthLayer;
	CONV_LAYER_1x1(5) sixthLayer;
	DENSE_LAYER(6) firstDenseLayer;
	DENSE_LAYER(7) secondDenseLayer;
#endif
};


