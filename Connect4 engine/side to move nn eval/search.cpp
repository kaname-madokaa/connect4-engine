#include "search.h"
#include <intrin.h>
#include <random>
#include <array>

//#define EVAL_NAIVE
//#define EVAL_MCTS
//#define EVAL_NN
std::random_device device;
std::mt19937_64 generator(device());

namespace search
{

	Score mctsPlaythroughs(int playthroughs, Board& pos)
	{
		Board boardCopy = pos;
		std::random_device device;
		std::mt19937_64 generator(device());
		std::uniform_int_distribution<int> distribution(0, 6);
		int totalScore = 0;
		for (int game = 0; game < playthroughs; game++)
		{
			while (pos.gameResult() == COLOR_NONE)
			{
				pos.doMove(distribution(generator));
			}
			if (pos.gameResult() == COLOR_YELLOW)
			{
				totalScore++;
			}
			else if (pos.gameResult() == COLOR_RED)
			{
				totalScore--;
			}
			pos = boardCopy;
		}
		return (Score)totalScore;
	}

	void init()
	{
		hashRandom;
		std::random_device device;
		std::mt19937_64 generator(device());
		for (int color = COLOR_YELLOW; color < COLOR_NONE; color++)
		{
			for (int col = 0; col < Board::cols; col++)
			{
				for (int row = 0; row < Board::rows; row++)
				{
					hashRandom[color][row][col] = generator();
				}
			}
		}
	}

	bool probeEntry(unsigned long long key, Score& score, Score alpha, Score beta, int depth)
	{
		if (hashMap.find(key) != hashMap.end())
		{
			if (depth == hashMap[key].depth)
			{
				if (hashMap[key].flag == FLAG_EXACT)
				{
					score = hashMap[key].score;
					return true;
				}
				else if (hashMap[key].flag == FLAG_ALPHA && hashMap[key].score <= alpha)
				{
					score = alpha;
					return true;
				}
				else if (hashMap[key].flag == FLAG_BETA && hashMap[key].score >= beta)
				{
					score = beta;
					return true;
				}
			}
		}
		return false;
	}

	Score evaluate(Board& pos, double noise, NN& net)
	{
		int totalScore = SCORE_DRAW;
		if (pos.gameResult() != COLOR_NONE)
		{
			//if (pos.gameResult() == COLOR_DRAW) return SCORE_DRAW;
			//return (Score)(SCORE_LOSS + pos.ply());
		}
#ifdef EVAL_MCTS
		totalScore = mctsPlaythroughs(50, pos);
#endif

#ifdef EVAL_NAIVE
		for (int col = 0; col < Board::cols; col++)
		{
			for (int row = 0; row < Board::rows; row++)
			{
				if (pos[COLOR_YELLOW] & (1ULL << (row * Board::cols + col)))
				{
					totalScore += (Board::rows - row) + (Board::cols - std::abs(3 - col));
				}
				else if (pos[COLOR_RED] & (1ULL << (row * Board::cols + col)))
				{
					totalScore -= (Board::rows - row) + (Board::cols - std::abs(3 - col));
				}
			}
		}
#endif

#ifdef EVAL_NN
		totalScore = net.calculateOutput(pos).sum() * 100;
#endif

#ifdef EVAL_NEAT

#endif
		if (noise != 0.00)
		{
			std::random_device device;
			std::uniform_int_distribution<int> dist(-noise * 100, noise * 100);
			std::mt19937_64 generator(device());
			totalScore += dist(generator);
		}
		return (Score)(totalScore);// *(pos.side() == COLOR_YELLOW ? 1 : -1));
	}
	unsigned long long hash(const Board& pos)
	{
		unsigned long long result{ 0 };
		for (int col = 0; col < Board::cols; col++)
		{
			for (int row = 0; row < Board::rows; row++)
			{
				if (pos[COLOR_YELLOW] & (1ULL << (row * Board::cols + col)))
				{
					result ^= hashRandom[COLOR_YELLOW][row][col];
				}
				else if (pos[COLOR_RED] & (1ULL << (row * Board::cols + col)))
				{
					result ^= hashRandom[COLOR_RED][row][col];
				}
			}
		}
		return result;
	}
	std::array<std::pair<Score, int>, 7> orderMoves(Board& pos, unsigned long long key)
	{
		std::array<std::pair<Score, int>, 7> moves;
		if (hashMap.find(key) != hashMap.end() || true)
		{
			std::array<std::pair<Score, int>,7> a =
			{
				 {{SCORE_INF, 3},{SCORE_INF, 4},{SCORE_INF, 2},{SCORE_INF, 5},{SCORE_INF, 1},{SCORE_INF, 6},{SCORE_INF, 0}}
			};
			return a;
		}
		/*
		int i = 0;
		if (hashMap[key].move != -1)
		{
			moves[0] = { SCORE_INF, hashMap[key].move };
			i++;
		}
		for (i; i < 7; i++)
		{
			if (pos.doMove(i))
			{
				unsigned long long key = hash(pos);
				moves[i] = { hashMap[key].score ,i };
				pos.undoMove();
			}
			else
			{
				moves[i] = { (Score)-SCORE_INF ,i };                                                                                                   
			}
		}
		std::sort(moves.begin(), moves.end(), std::greater<>());
		*/
		return moves;
	}
	Score search(Board& pos, int depth, Score alpha, Score beta, double noise)
	{
		if (depth == 0 || pos.gameResult() != COLOR_NONE)
		{
			return (Score)(evaluate(pos, noise));
		}
		unsigned long long key = hash(pos);
		Score oldAlpha = alpha;	
		Score bestScore = (Score)-SCORE_INF;
		Score currScore = bestScore;


		if (probeEntry(key, currScore, alpha, beta, depth))
		{
			return currScore;
		}
		int bestMove = 3;
		std::array<std::pair<Score, int>, 7> moves = orderMoves(pos, key);
		for (auto& move : moves)
		{
			int& i = move.second;
			if (pos.doMove(i))
			{
				if(i == moves[0].second)
					currScore = (Score)-search(pos, depth - 1, (Score)-beta, (Score)-alpha, noise);
				else
				{
					currScore = (Score)-search(pos, depth - 1, (Score)(-alpha - 1), (Score)-alpha, noise);
					if (alpha < currScore && currScore < beta)
					{
						currScore = (Score)-search(pos, depth - 1, (Score)(-beta), (Score)-currScore, noise);
					}
				}
				pos.undoMove();
				if (bestScore < currScore)
				{
					bestScore = currScore;
					bestMove = i;
				}
				alpha = std::max(alpha, bestScore);
				if (beta <= alpha)
				{
					hashMap[key] = { beta, depth, bestMove, FLAG_BETA };
					return bestScore;
				}
			}
		}

		if (alpha != oldAlpha)
		{
			hashMap[key] = { bestScore, depth, bestMove, FLAG_EXACT };
		}
		else
		{
			hashMap[key] = { bestScore, alpha, bestMove, FLAG_ALPHA };
		}

		return alpha;
	}




}
