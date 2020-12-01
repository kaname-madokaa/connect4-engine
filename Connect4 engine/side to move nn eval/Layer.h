#pragma once
#include <vector>
#include <Eigen/Dense>



class Layer
{
public:
	virtual const Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) = 0;
	virtual char layerType() = 0;
	virtual void printWeights() = 0;
};

