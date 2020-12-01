#pragma once
#include <string>
#include <iostream>
#include "Board.h"
#include "search.h"
#include <cstdlib>
#include <chrono>
#include <thread>
#include <mutex>

//#define NN_VALUE
//#define NN_POLICY

#ifdef NN_POLICY
static Eigen::MatrixXf currentOutput;
#endif

class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using clock_t = std::chrono::high_resolution_clock;
	using second_t = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<clock_t> m_beg;

public:
	Timer() : m_beg(clock_t::now())
	{
	}

	void reset()
	{
		m_beg = clock_t::now();
	}

	double elapsed() const
	{
		return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count();
	}
};

//it's so convienient that universal chess interface and universal connect4 interface share the same abbrevation lol
namespace uci
{
	using namespace std::literals::string_literals;
	void generateWeights();
	void selfplay(int numGames, int threads);
	void rate();
	void calculate();
	inline std::random_device device;
	inline std::mt19937 generator(device());
	inline std::uniform_int_distribution<int> chance(0, 100);
	inline std::uniform_int_distribution<int> abcd(0, 6);
	inline std::map<unsigned long long, int> NN_eval_cache;
	std::mutex mu;

	void position(Board& pos, const std::string& command)
	{
		pos.newGame();

		if (command.find("startpos") != std::string::npos)
		{
			//we iterate through all the moves given to us in a position command
			//we assume that the gui ensures the validity of the moves
			//which it does HA!
			for (int character = 18; character < command.size(); character++)
			{
				pos.doMove(command[character] - '0');
			}
		}
		//to do: implement a "fen" position setup
	}

	int go(Board& pos, double noise = 0.00, bool output = true, int depth = 0)
	{
		int bestMoveIndex = 0;
#ifndef NN_POLICY
		Score bestMoveValue = (Score) -SCORE_INF;

		Score alpha = (Score)-SCORE_INF;
		Score beta = SCORE_INF;
		for (auto move : { 3,4,2,5,1,6,0 })
		{
			if (pos.doMove(move))
			{
				Score currVal;
				if (pos.gameResult() != COLOR_NONE)
				{
					//if (pos.gameResult() == COLOR_DRAW) currVal = SCORE_DRAW;
					//else
					//	currVal = (Score)(SCORE_WIN - pos.ply());
					currVal = (Score)-search::search(pos, 0, (Score)-beta, (Score)-alpha, noise);
				}
				else
				{
					currVal = (Score)-search::search(pos, depth, (Score)-beta, (Score)-alpha, noise);
				}
				pos.undoMove();
				if (bestMoveValue < currVal)
				{
					bestMoveIndex = move;
					bestMoveValue = currVal;
				}
				alpha = std::max(alpha, bestMoveValue);
				if (beta <= alpha)
				{
					break;
				}
			}
		}
#endif
#ifdef NN_POLICY
		float bestMoveValue = -100.0f;
		currentOutput = network.calculateOutput(pos);
		for (int move = 0; move < Board::cols; move++)
		{
			if (pos.doMove(move) && currentOutput(move, 1) > bestMoveValue)
			{
				pos.undoMove();
				bestMoveIndex = move;
				bestMoveValue = currentOutput(move, 1);
			}
		}
#endif
		if (output)
		{
			std::cout << "bestmove: " << bestMoveIndex << "\n";
			std::cout << "score: " << bestMoveValue << "\n";
		}
		hashMap.clear();
		return bestMoveIndex;
	}

	void Loop()
	{
		Board rootPos;
		std::string command;
		while (true)
		{
			std::getline(std::cin, command);
			if (command.find("go") != std::string::npos)
			{
				//while (true)
				{
					Timer a;
					for (int depth = 2; depth < 8; depth++)
					{
					}
					go(rootPos, 0.0, true, 0);
					std::cout << a.elapsed() << "\n";
					//hashMap.clear();
				}

			}
			else if (command.find("position") != std::string::npos) position(rootPos, command);
			else if (command.find("print") != std::string::npos) rootPos.printBoard();
			else if (command.find("generate") != std::string::npos) generateWeights();
			else if (command.find("selfplay") != std::string::npos)
			{
				std::stringstream ss(command.substr(8));
				int numGames;
				ss >> numGames;
				selfplay(numGames,3);
			}
			else if (command.find("scan") != std::string::npos) system("tree");
			else if (command.find("eval") != std::string::npos) std::cout << search::evaluate(rootPos) << "\n";
			else if (command.find("rate") != std::string::npos) while(1) rate();
			else if (command.find("calculate") != std::string::npos) calculate();
			
#ifdef _DEBUG
			else if (command.find("test") != std::string::npos)
			{
				rootPos.flipSideToMove();
				rootPos.isOver();
				rootPos.flipSideToMove();
				//rootPos.doMove(command[5] - '0');
			}
#endif
			else
			{
				std::cout << '\'' << command.substr(0, command.find(' ')) << '\'' << " is not recognized as an internal or external command,\noperable program or batch file.\n";
			}
			
		}
	}



void generateWeights()
{

}

struct Game
{
	std::vector<Eigen::MatrixXf> inputs;
	std::vector<int> moves;
	float result;
#ifdef NN_POLICY
	std::vector<Eigen::MatrixXf> answers;
#endif
	Game& operator=(Game& second)
	{
		inputs = second.inputs;
		moves = second.moves;
		result = second.result;
		return *this;
	}
	Game(Game& second)
	{
		*this = second;
	}
	Game(Game&& second) noexcept
	{
		*this = std::move(second);
	}
	Game()
	{

	}
	
	Game& operator=(Game&& second) noexcept
	{
		inputs.reserve(second.inputs.size());
		for (int input = 0; input < second.inputs.size(); input++)
		{
			inputs.emplace_back(std::move(second.inputs[input]));
		}
		moves = std::move(second.moves);
		result = second.result;
		return *this;
	}

	void saveToFile();
};

int evalCacheHits = 0;
int evalCacheMisses = 0;

int NN_go(Board& pos, double noise, NN& net = network)
{
	Score bestScore = (Score)-1000000;
	int bestMove = 0;

	std::uniform_int_distribution<int> noise_dist(-noise * 100, noise * 100);
	for (int move = 0; move < 7; move++)
	{
		if (pos.doMove(move))
		{
			Score score;
			unsigned long long posKey = search::hash(pos);
			if (NN_eval_cache.find(posKey) != NN_eval_cache.end())
			{
				score = (Score) NN_eval_cache[posKey];
				mu.lock();
				evalCacheHits++;
				mu.unlock();
			} 
			else
			{
				score = (Score)-search::evaluate(pos, 0.0, net);
				mu.lock();
				NN_eval_cache[posKey] = score;
				evalCacheMisses++;
				mu.unlock();
			}
			mu.lock();
			score = (Score)(score + noise_dist(generator));
			mu.unlock();
			if (score > bestScore)
			{
				bestScore = score;
				bestMove = move;
			}
			pos.undoMove();
		}
	}
	return bestMove;
}

void playGames(int numGames, std::vector<Game>& games, std::vector<float>& results, int& numPositions, NN& net = network)
{
	Board pos;
	std::uniform_int_distribution<int> randomMove(0,6);
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_int_distribution<int> chance(0, 100);

	for (int gameNum = 0; gameNum < numGames; gameNum++)
	{
		Game currGame;
		int moveNum = 0;
		currGame.inputs.emplace_back(network.boardToInput(pos));
		mu.lock();
		numPositions++;
		mu.unlock();
		double noise;
		while (pos.gameResult() == COLOR_NONE)
		{
			//std::cout << NN::boardToInput(pos).transpose() << "\n\n";
			//std::cout << pos.ply() << "\n";
			int move;
			noise = 0;
			if ((chance(generator) < 25) || (pos.ply() > 14 && (chance(generator) < 35)))
			{
				noise = 1.0;
			}
			move = NN_go(pos, noise, net);
			
			if (pos.doMove(move))
			{
				currGame.inputs.emplace_back(network.boardToInput(pos));
#ifdef NN_POLICY
				currGame.answers.emplace_back(std::move(currentOutput));
#endif
				currGame.moves.emplace_back(move);
				mu.lock();
				numPositions++;
				mu.unlock();
				moveNum++;
			}
		}
		currGame.inputs.emplace_back(pos.get_opposite_NN_input());
		mu.lock();
		numPositions++;
		mu.unlock();
		mu.lock();
#ifndef NN_POLICY
		if (pos.gameResult() == COLOR_YELLOW) results.push_back(1.0);
		else if (pos.gameResult() == COLOR_RED) results.push_back(-1.0);
		else
		{
			results.push_back(0.0);
		}
#endif
#ifdef NN_POLICY
		for (int move = 0; move < currGame.moves.size(); move++)
		{
			currGame.answers[move](currGame.moves[move], 1) = pos.gameResult() == COLOR_YELLOW ? 1.0f : pos.gameResult() == COLOR_DRAW ? 0 : -1.0f;
		}
#endif
		//std::cout << currGame.result << "\n";
		if (!(gameNum % (numGames / 10)))
		{
			std::cout << "finished " << gameNum << "/" << numGames << " games\n";
		}
		games.emplace_back(std::move(currGame));
		mu.unlock();
		pos.newGame();
	}
}

void selfplay(int numGames, int threads)
{
#ifndef NN_POLICY
	int batches = numGames / 2000;
	int gamesPerBatch = numGames / batches;
	std::vector<NN> networks(threads);
	for (auto& net : networks)
	{
		net = NN("C:/Users/Anastazja/abcd");
	}
	for (int batch = 0; batch < batches; batch++)
	{
		std::vector<float> results;
		std::cout << "batch " << batch << " out of " << batches << "\n";
		results.reserve(gamesPerBatch);
#endif
#ifdef NN_POLICY
		std::vector<Eigen::MatrixXf> answers;
#endif
		std::vector<Game> games;
		games.reserve(gamesPerBatch);
		int numPositions = 0;

		std::vector<std::thread> thread_pool;
		int gamesPerThread = gamesPerBatch / threads;

		for (int thread = 0; thread < threads; thread++)
		{
			thread_pool.emplace_back(std::thread(playGames, gamesPerThread, std::ref(games), std::ref(results), std::ref(numPositions), std::ref(networks[thread])));
		}
		for (auto& thread : thread_pool)
		{
			thread.join();
		}
		std::cout << "evalCacheMisses " << evalCacheMisses << "     evalCacheHits " << evalCacheHits << "\n";
		evalCacheMisses = 0;
		evalCacheHits = 0;
		std::cout << "finished " << numGames << " selfplay games!\n";
		std::ofstream output("games "s + std::to_string(batch)+ ".bin" , std::ios::binary);
		int posNum = 0;
		output.write((char*)&numPositions, sizeof(int));
		for (int gameNum = 0; gameNum < gamesPerBatch; gameNum++)
		{
			for (int moveNum = 0; moveNum < games[gameNum].inputs.size(); moveNum++)
			{
				//std::cout << game.inputs[move] << "\n\n";
				// output << ((game.result == COLOR_YELLOW) ? "Y " : (game.result == COLOR_DRAW ? "D " : "R "));
				posNum++;
				for (int row = 0; row < Board::rows; row++)
				{
					for (int col = 0; col < Board::cols; col++)
					{


						//


						for (int channel = 0; channel < input_channels; channel++)
						{
							output.write((char*)&games[gameNum].inputs[moveNum](row, col + Board::cols * channel), sizeof(float));
						}

					}

				}



			}
		}
		//std::cout << games[0].inputs[0];
		for (int gameNum = 0; gameNum < gamesPerBatch; gameNum++)
		{
			for (int moveNum = 0; moveNum < games[gameNum].inputs.size(); moveNum++)
			{
				posNum++;
				// output << ((game.result == COLOR_YELLOW) ? "Y " : (game.result == COLOR_DRAW ? "D " : "R "));
				//std::cout << results[gameNum] << "\n";
				//std::cout << inputs[gameNum * Board::cols * Board::rows+ moveNum] << "\n\n";
				output.write((char*)&results[gameNum], sizeof(float));	
				results[gameNum] *= -1;
			}
		}
		std::cout << posNum << "\n";
		numPositions = 0;
		output.close();
	}
	
	exit(1);
}



void rate()
{
	network = NN("C:/Users/Anastazja/abcd");
	constexpr int numGames = 1000;
	std::random_device device;
	std::mt19937_64 generator(device());
	std::uniform_int_distribution<int> abcd(0, 6);
	double score = 0;
	int wins = 0;
	int draws = 0;
	int loses = 0;
	for (int game = 0; game < numGames; game++)
	{
		Board pos;
		while (pos.gameResult() == COLOR_NONE)
		{
			int move;
			if ((pos.side() == COLOR_YELLOW && (game % 2)) || (pos.side() == COLOR_RED && !(game % 2)))
			{
				while (!pos.doMove(abcd(generator)));
			}
			else
			{
				move = NN_go(pos, 0.0);
				pos.doMove(move);
			}
		}
		if ((pos.gameResult() == COLOR_YELLOW && (game % 2)) || (pos.gameResult() == COLOR_RED && !(game % 2))) 
		{
			score -= 1.0; loses++;
		}
		else if ((pos.gameResult() == COLOR_RED && (game % 2)) || (pos.gameResult() == COLOR_YELLOW && !(game % 2)))
		{
			score += 1.0; wins++;
		}
		else
		{
			draws++;
		}
	}
	std::cout << "score: " << score << "\n";
	std::cout << "W/D/L: " << wins << "/" << draws << "/" << loses << "\n";
	exit(1);
}

void calculate()
{
	std::ifstream output("games.bin", std::ios::binary);
	int numPositions;
	output.read((char*)&numPositions, sizeof(int));
	for (int pos = 0; pos < 100; pos++)
	{
		Eigen::MatrixXf posInput(6, 7 * input_channels);
		for (int row = 0; row < Board::rows; row++)
		{
			for (int col = 0; col < Board::cols; col++)
			{
				for (int channel = 0; channel < input_channels; channel++)
				{
					output.read((char*)&posInput(row, col + Board::cols * channel), sizeof(float));
				}

			}
		}
		std::cout << posInput << ":" << "\n";
		std::cout<<network.calculateOutput(posInput) << "\n";
	}
}

}