#pragma once
#include <exception>
#include <functional>
#include <array>
#include <vector>
#include <deque>
#include <Eigen/Dense>

enum Color
{
	COLOR_YELLOW,
	COLOR_RED,
	COLOR_NONE,
	COLOR_DRAW  //used for result
};


struct Point 
{
	unsigned int row;
	unsigned int col;
};


//bitboard representation, it assumes that the playing board is not bigger than 64 fields in total (default is 6x7 which is 42 fields)
class Board
{
public:
	static constexpr unsigned long long rows{ 6 }; //const as fuck
	static constexpr unsigned long long cols{ 7 }; //won't ever change

	Board();
	const unsigned long long& operator[](int index) const;
	bool doMove(int index);
	void undoMove();
	void printBoard();
	void newGame();

	Color side() const { return sideToMove; }
	Color gameResult() const { return result; }
	int ply() const { return gamePly; }
	const std::vector<Point>& getMoveHistory() const { return moveHistory; }
#if EVAL_NN || POLICY_NN
	const Eigen::MatrixXf& get_NN_input() const { return sideToMove == COLOR_YELLOW? m_input_yellow: m_input_red; }
#endif
	unsigned long long columnBitmasks[cols];
	unsigned long long rowBitmasks[rows];

#ifdef NDEBUG
private:
#endif
#ifdef _DEBUG
	void flipSideToMove()
	{
		sideToMove ? sideToMove = COLOR_YELLOW : sideToMove = COLOR_RED;
	}
#endif

	void isOver();
	static const unsigned long long verticalVictoryBitmask = 1 | (1 << 7) | (1 << 14) | (1 << 21);
	static const unsigned long long horizontalVictoryBitmask = 1 | (1 << 1) | (1 << 2) | (1 << 3);
	static const unsigned long long diagonalRightVictoryBitmask = ((1<<3) | (1 << 9) | (1 << 15) | (1 << 21))>>3;
	static const unsigned long long diagonalLeftVictoryBitmask = 1 | (1 << 8) | (1 << 16) | (1 << 24);

	unsigned long long pieceBitboards[COLOR_NONE];

	std::array<unsigned int, cols> columnSizes;
	std::vector<Point> moveHistory;

	Color sideToMove;
	Color result;
	bool reachedMinimalHeight;
	int gamePly;


#if EVAL_NN || POLICY_NN
	//input matrix of dimensions rows * cols * input channels used in CNN 
	Eigen::MatrixXf m_input_yellow;
	Eigen::MatrixXf m_input_red;
#endif

#if _NEAT
	//input used by the NEAT algorithm
	std::array<float, Board::rows* Board::cols * 2> m_input;
#endif
};

