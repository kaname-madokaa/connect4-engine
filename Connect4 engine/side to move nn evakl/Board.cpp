#include "Board.h"
#include <intrin.h>
#include <iostream>
#include "NN.h"


Board::Board() :pieceBitboards{ 0,0 }, result{ COLOR_NONE }, gamePly{ 0 }, reachedMinimalHeight{ false }
{
	moveHistory.reserve(42);
	columnSizes.fill(0);
	sideToMove = COLOR_YELLOW;
	//prepare the column and row bitmasks
	for (int row = 0; row < rows; row++)
	{
		unsigned long long currRow = 0;
		for (int col = 0; col < cols; col++)
		{
			currRow |= (1ull << col);
		}
		rowBitmasks[row] = currRow << (row * cols);
	}

	for (int col = 0; col < cols; col++)
	{
		unsigned long long currCol = 0;
		for (int row = 0; row < rows; row++)
		{
			currCol |= (1ull << (cols * row));
		}
		columnBitmasks[col] = currCol << col;
	}
#if EVAL_NN || POLICY_NN
	m_input_yellow.resize(Board::rows, Board::cols * input_channels);
	m_input_yellow.setZero();
	m_input_red.resize(Board::rows, Board::cols * input_channels);
	m_input_red.setZero();
#endif
}


void Board::printBoard()
{
	std::cout << "\n";
	for (int row = rows-1; row >=0; row--)
	{
		for (int col = cols-1 ; col >=0 ; col--)
		{
			unsigned long long bitmask = 1ULL << (col + row * cols);
			std::cout << "[";
			if (pieceBitboards[COLOR_YELLOW] & bitmask)
			{
				std::cout << "O";
			}
			else if (pieceBitboards[COLOR_RED] & bitmask)
			{
				std::cout << "X";
			}
			else
			{
				std::cout << " ";
			}
			std::cout << "]";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}


const unsigned long long& Board::operator[](int index) const
{
	if (index < COLOR_NONE && index>=0)
	{
		return pieceBitboards[index];
	}
	else
	{
		new int[10000];	//punishment for being an awful programmer
		std::cout << "illegal access detected" << "\n";
		throw std::exception("fuck");
	}
}

bool Board::doMove(int index)
{
	if (index >= 7 || index < 0)
	{
		return false;
	}
	else if (columnSizes[index] >= rows)
	{
		return false;
	}
	else if (result == COLOR_NONE)
	{
		gamePly++;
		unsigned long long& currentSideBitboard = pieceBitboards[sideToMove];
		moveHistory.push_back({ columnSizes[index], (unsigned)index });
		currentSideBitboard |= (1ull << (index + columnSizes[index]++ * cols));
		if (columnSizes[index] >= 4)
		{
			reachedMinimalHeight = true;
		}
		isOver();
#if EVAL_NN || POLICY_NN

		int moveHistoryIndex = moveHistory.size() - 1;
		for (int channel = 0; channel < (input_channels/3) && moveHistoryIndex >= 0; channel++, moveHistoryIndex--)
		{
			const Point& move = moveHistory[moveHistoryIndex];

			//put current move in the "both pieces" channel
			m_input_yellow(move.row, move.col + (channel * cols * 3) + cols * 2) = 1;
			m_input_red(move.row, move.col + (channel * cols * 3) + cols * 2) = 1;
			if (sideToMove == COLOR_YELLOW)
			{
				//current move is yellow's
				m_input_yellow(move.row, move.col + (channel * cols * 3)) = 1;
				//current move is not red's
				m_input_red(move.row, move.col + (channel * cols * 3) + cols) = 1;
			}
			else
			{
				//current move is red's
				m_input_red(move.row, move.col + (channel * cols * 3)) = 1;
				//current move is not yellow's
				m_input_yellow(move.row, move.col + (channel * cols * 3) + cols) = 1;
			}
		}
#endif
		sideToMove ? sideToMove = COLOR_YELLOW : sideToMove = COLOR_RED;
		return true;
	}
	else
	{
		new int[10000];	//punishment for being an awful programmer
		std::cout << "illegal move detected" << "\n";
		//throw std::exception("fuck");
	}
}

//this function checks if the previous move changed the current result of the game
void Board::isOver()
{
	const Point& lastMove = moveHistory.back();
	const unsigned long long& currentSideBitboard = pieceBitboards[sideToMove];



	//check for the vertical victories, those are only possible if we have at least 4 pieces in that column
	if (__popcnt64(currentSideBitboard & columnBitmasks[lastMove.col]) >= 4)
	{
		unsigned long long bitMask = verticalVictoryBitmask << lastMove.col;
		//since the bitmask is 4 rows tall, we will look at the current row and 3 rows above it, meaning we can (and should) end the loop early
		for (int row = 0; row < rows - 3; row++)
		{
			//we first isolate the bitmask bits and then check if all of them are 1, if all are 1 then xor with mask should zero them out
			if (!((currentSideBitboard & bitMask) ^ bitMask))
			{
				//we found a victory pattern, meaning the last move ended the game
				result = sideToMove;
				return;
			}
			//move the bitmask one row higher (each row has "cols" cells)
			bitMask <<= cols;
		}
	}

	//check for the horizontal victories, those are only possible if we have at least 4 pieces in that row
	if (__popcnt64(currentSideBitboard & rowBitmasks[lastMove.row]) >= 4)
	{
		//row * cols puts the bitmask in the correct row
		unsigned long long bitMask = horizontalVictoryBitmask << (lastMove.row * cols);
		//since the bitmask is 4 columns wide, we will look at the current column and 3 next to it, meaning we can (and should) end the loop early
		for (int col = 0; col < cols - 3; col++)
		{
			//we first isolate the bitmask bits and then check if all of them are 1, if all are 1 then xor with mask should zero them out
			if (!((currentSideBitboard & bitMask) ^ bitMask))
			{
				//we found a victory pattern, meaning the last move ended the game
				result = sideToMove;
				return;
			}
			//move the bitmask one column to the left
			bitMask <<= 1;
		}
	}

	//check for diagonal victories
	//it's impossible to create a diagonal for all positions on the board, which means that we first need to check if a diagonal is possible to exist.
	//if it is proven to be possible, we can just calculate which diagonal last move belongs to and do the same thing we did with rows and columns

	//check if any diagonal is possible
	//no diagonal is possible to exist if:
	//-no column has reached the height of 4
	//-we are before move 10 of the game
	if (!reachedMinimalHeight || (gamePly < 10))
	{
		return;
	}

	//check if left diagonals are possible
	//those cannot be formed in left bottom corner and in top right corner
	if (!(
		(
			(lastMove.row < 3 && lastMove.col > cols - 2) ||
			(lastMove.row < 2 && lastMove.col > cols - 3) ||
			(lastMove.row < 1 && lastMove.col > cols - 4)
			)
		||
		(
			(lastMove.row > rows - 4 && lastMove.col < 1) ||
			(lastMove.row > rows - 3 && lastMove.col < 2) ||
			(lastMove.row > rows - 2 && lastMove.col < 3)
			)
		))
	{
		//diagonals are possible, check if one has indeed been formed
		//begining point of the diagonal lastMove belongs to has coordinates:
		//x = smaller of (row,col)
		//coords: (row - x), (col - x)
		//we can do better however, as our last move will impact fields at most 3 rows and columns away 
		//meaning that if x is bigger than 3, we need not subtract more than 3
		//it also means that we can stop the loop early, as from for example 0,0, we won't affect further than 3,3

		int x = std::min(lastMove.row, lastMove.col);
		const int beginRow = lastMove.row - (x > 3 ? 3 : x);
		const int beginCol = lastMove.col - (x > 3 ? 3 : x);

		//now the end coordinates
		//those normally would be equivalent to:
		//x = smaller of (rows - row, cols - col)
		//coords: (row + x), (col + x)
		//however if x happens to be greater than 3, we need not look that far away as we wouldn't have impacted those fields anyways

		x = std::min(rows - lastMove.row, cols - lastMove.col);

		const int endRow = lastMove.row + (x > 3 ? 3 : x);
		const int endCol = lastMove.col + (x > 3 ? 3 : x);



		//we're now ready to do the loop!

		//put the mask in the begining position 
		unsigned long long bitMask = diagonalLeftVictoryBitmask << (beginRow * cols + beginCol);
		//as always, mask starts in the current row and extends 3 fields further
		for (int row = beginRow, col = beginCol; row + 3 < rows && col + 3 < cols && row + 3 <= endRow && col + 3 <= endCol; row++, col++)
		{
			if (!((currentSideBitboard & bitMask) ^ bitMask))
			{
				//we found a victory pattern, meaning the last move ended the game
				result = sideToMove;
				return;
			}
			bitMask <<= cols + 1; //move it one row up, one column left
		}
	}



	//check if right diagonals are possible
	//those cannot be formed in right bottom corner and in top left corner
	if (!(
		(
			(lastMove.col < 3 && lastMove.row < 1) ||
			(lastMove.col < 2 && lastMove.row < 2) ||
			(lastMove.col < 1 && lastMove.row < 3)
		)
		||
		(
			(lastMove.row > rows - 4 && lastMove.col > cols - 2) ||
			(lastMove.row > rows - 3 && lastMove.col > cols - 3) ||
			(lastMove.row > rows - 2 && lastMove.col > cols - 4)
		)
		))
	{
		//diagonals are possible, check if one has indeed been formed
		//begining point of the diagonal lastMove belongs to has coordinates:
		//x = smaller of (row, cols - col)
		//coords: (row - x), (col + x)
		//we can do better however, as our last move will impact fields at most 3 rows and columns away 
		//meaning that if x is bigger than 3, we need not subtract more than 3
		//it also means that we can stop the loop early, as from for example 0,4, we won't affect further than 3,1
		//we subtract 1 from x in case we begin in an invalid index column

		int x = std::min(lastMove.row - 0ull, cols - lastMove.col - 1);
		const int beginRow = lastMove.row - (x > 3 ? 3 : x);
		const int beginCol = lastMove.col + (x > 3 ? 3 : x);

		//a check to keep things in bounds after additions, as it can reach column equal to cols which breaks everything


		//now the end coordinates
		//those normally would be equivalent to:
		//x = smaller of (rows - row, col)
		//coords: (row + x), (col - x)
		//however if x happens to be greater than 3, we need not look that far away as we wouldn't have impacted those fields anyways

		x = std::min(rows - lastMove.row, cols);

		const int endRow = lastMove.row + (x > 3 ? 3 : x);
		const int endCol = lastMove.col - (x > 3 ? 3 : x);



		//we're now ready to do the loop!

		//put the mask in the begining position 
		unsigned long long bitMask = diagonalRightVictoryBitmask << (beginRow * cols + beginCol);
		//as always, mask starts in the current row and extends 3 fields further
		for (int row = beginRow, col = beginCol; row + 3 < rows && col - 3 >=0 && row + 3 <= endRow && col - 3 >= endCol; row++, col--)
		{
			if (!((currentSideBitboard & bitMask) ^ bitMask))
			{
				//we found a victory pattern, meaning the last move ended the game
				result = sideToMove;
				return;
			}
			bitMask <<= cols - 1; //move it one row up, one column right
		}
	}
	if (gamePly == 42)
	{
		result = COLOR_DRAW;
		return;
	}
	
}


void Board::undoMove()
{
	const Point& lastMove = moveHistory.back();
	//bring back the side to move
	sideToMove ? sideToMove = COLOR_YELLOW : sideToMove = COLOR_RED;	
	//if last move won the game, cancel the win, otherwise leave it as it was
	result = COLOR_NONE;	
	//remove one from the column
	columnSizes[lastMove.col]--; 


	//we check if we can uncheck the minimal height property
	if (reachedMinimalHeight)
	{
		reachedMinimalHeight = false;
		for (const auto& val : columnSizes)
		{
			if (val >= 4)
			{
				reachedMinimalHeight = true;
				break;
			}
		}
	}

	//flip the previously set bit back to 0
	unsigned long long& currentSideBitboard = pieceBitboards[sideToMove];
	unsigned long long bit = 1ULL << (lastMove.row * cols + lastMove.col);
	currentSideBitboard ^= bit; //xor flips the bit

	//decrement move counter
	gamePly--;

	//bring back the previous input
#if EVAL_NN || POLICY_NN
	int moveHistoryIndex = moveHistory.size() - 1;
	for (int channel = 0; channel < (input_channels / 3) && moveHistoryIndex >= 0; channel++, moveHistoryIndex--)
	{
		const Point& move = moveHistory[moveHistoryIndex];
		m_input_yellow(move.row, move.col + (channel * cols * 3) + cols * 2) = 0;
		m_input_red(move.row, move.col + (channel * cols * 3) + cols * 2) = 0;
		if (sideToMove == COLOR_YELLOW)
		{
			//current move is yellow's
			m_input_yellow(move.row, move.col + (channel * cols * 3)) = 0;
			//current move is not red's
			m_input_red(move.row, move.col + (channel * cols * 3) + cols) = 0;
		}
		else
		{
			//current move is red's
			m_input_red(move.row, move.col + (channel * cols * 3)) = 0;
			//current move is not yellow's
			m_input_yellow(move.row, move.col + (channel * cols * 3) + cols) = 0;
		}
	}
#endif

	//finally, remove the move from history 
	moveHistory.pop_back();
}


void Board::newGame()
{
	moveHistory.clear();
	pieceBitboards[COLOR_YELLOW] = 0;
	pieceBitboards[COLOR_RED] = 0;
	result = COLOR_NONE;
	gamePly = 0;
	reachedMinimalHeight = false;
	moveHistory.reserve(42);
	columnSizes.fill(0);
	sideToMove = COLOR_YELLOW;

#if EVAL_NN || POLICY_NN
	m_input_yellow.resize(Board::rows, Board::cols * input_channels);
	m_input_yellow.setZero();
	m_input_red.resize(Board::rows, Board::cols * input_channels);
	m_input_red.setZero();
#endif
}