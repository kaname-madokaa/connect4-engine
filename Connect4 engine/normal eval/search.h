#pragma once
#include "Board.h"
#include "NN.h"
#include <map>



enum Score
{
	SCORE_LOSS = -10000,
	SCORE_DRAW = 0,
	SCORE_WIN = 10000,
	SCORE_INF = 100000
};

enum TTflag
{
	FLAG_NONE,
	FLAG_ALPHA,
	FLAG_BETA,
	FLAG_EXACT
};

struct gameInfo
{
	Score score;
	int depth;
	int move=-1;
	TTflag flag;
};

inline std::map<unsigned long long, gameInfo> hashMap;
inline unsigned long long hashRandom[COLOR_NONE][Board::rows][Board::cols];
inline NN network("C:/Users/Anastazja/abcd");

namespace search
{
	Score search(Board& pos, int depth, Score alpha, Score beta, double noise = 0.00);
	unsigned long long hash(const Board& pos);
	void init();
	Score evaluate(Board& pos, double noise = 0.00, NN& net = network);
}
