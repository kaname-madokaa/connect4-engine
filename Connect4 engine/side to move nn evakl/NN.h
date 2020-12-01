#pragma once
#include "Board.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "ConvLayer.h"
#include "DenseLayer.h"

constexpr int minibatch_size = 100;
constexpr double dropoutChance = 0.3;
constexpr double learningRate = 0.1;
constexpr float epsilon = 0.00000001f;
constexpr int input_channels = 3;

class NN
{
public:
	NN() :NN("C:/Users/Anastazja/abcd") {}
	NN(std::string dir);
	NN(NN&& other) noexcept;
	NN& operator=(NN&& other) noexcept;
	Eigen::MatrixXf& calculateOutput(const Board& pos);
	Eigen::MatrixXf& calculateOutput(const Eigen::MatrixXf& input);
	const Eigen::MatrixXf& boardToInput(const Board& pos);

private:
	void dropout(Eigen::MatrixXf& input);
	std::vector<std::unique_ptr<Layer>> m_layers;
	int lastRecordedPly;
};


