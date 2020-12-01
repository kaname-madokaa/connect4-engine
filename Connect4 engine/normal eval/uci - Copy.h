#pragma once
#include <string>
#include <iostream>
#include "Board.h"
#include "search.h"
#include <cstdlib>
#include <chrono>

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
	void generateWeights();
	void selfplay(int numGames);
	void rate();
	void calculate();
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
					go(rootPos, 0.10, true, 3);
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
				selfplay(numGames);
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
	std::ofstream file("abcd", std::ios::binary);
	int layersNum = 2;
	int layerSizes[] = { 30,1 };
	file.write((char*)&layersNum, 4);
	file.write((char*)layerSizes, 8);
	Eigen::MatrixXf firstLayerWeights;
	Eigen::MatrixXf secondLayerWeights;
	Eigen::MatrixXf firstLayerBiases;
	Eigen::MatrixXf secondLayerBiases;
	firstLayerWeights.resize(layerSizes[0], 84);
	secondLayerWeights.resize(layerSizes[1], layerSizes[0]);
	firstLayerBiases.resize(layerSizes[0], 1);
	secondLayerBiases.resize(layerSizes[1], 1);
	std::random_device device;
	std::mt19937_64 generator(device());
	std::uniform_real_distribution<double> abcd(-1, 1);

	for (auto& val : firstLayerWeights.reshaped()) val = abcd(generator);
	for (auto& val : secondLayerWeights.reshaped()) val = abcd(generator);
	for (auto& val : firstLayerBiases.reshaped()) val = abcd(generator);
	for (auto& val : secondLayerBiases.reshaped()) val = abcd(generator);

	file.write((char*)firstLayerWeights.data(), firstLayerWeights.size() * sizeof(float));
	file.write((char*)secondLayerWeights.data(), secondLayerWeights.size() * sizeof(float));

	file.write((char*)firstLayerBiases.data(), firstLayerBiases.size() * sizeof(float));
	file.write((char*)secondLayerBiases.data(), secondLayerBiases.size() * sizeof(float));

	file.close();
	network = NN("abcd");
	std::cout << "generated\n";
}

struct Game
{
	std::vector<Eigen::MatrixXf> inputs;
	std::vector<int> moves;
	float result;
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
	Game(Game&& second)
	{
		*this = std::move(second);
	}
	Game()
	{

	}
	
	Game& operator=(Game&& second)
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

void selfplay(int numGames)
{
	Board pos;
	int numPositions = 0;
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_int_distribution<int> abcd(0, 6);
	std::uniform_int_distribution<int> chance(0, 100);
	float* results = new float[numGames];
	std::vector<Game> games;

	for (int gameNum = 0; gameNum < numGames; gameNum++)
	{
		Game currGame;
		int moveNum = 0;
		currGame.inputs.emplace_back(std::move(NN::boardToInput(pos)));
		numPositions++;
		while (pos.gameResult() == COLOR_NONE)
		{
			int move;
			{
				{
					move = go(pos, 0.05, false, 0);
				}
			}
			if (pos.doMove(move))
			{
				currGame.inputs.emplace_back(std::move(NN::boardToInput(pos)));
				//currGame.moves.emplace_back(move);
				numPositions++;
				moveNum++;
			}
		}
		if (pos.gameResult() == COLOR_YELLOW) results[gameNum] = 1.0;
		else if (pos.gameResult() == COLOR_RED) results[gameNum] = -1.0;
		else
		{
			results[gameNum] = 0.0;
		}
		//std::cout << currGame.result << "\n";
		pos.newGame();
		if (!(gameNum % (numGames / 10)))
		{
			std::cout << "finished " << gameNum << "/" << numGames << " games\n";
		}
		games.emplace_back(std::move(currGame));

	}
	std::cout << "finished " << numGames << " selfplay games!\n";
	std::ofstream output("games.bin", std::ios::binary);
	int posNum = 0; 
	output.write((char*)&numPositions, sizeof(int));
	for (int gameNum = 0; gameNum < numGames; gameNum++)
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
						
							
								for (int channel = 0; channel < 12; channel++)
								{
									output.write((char*)&games[gameNum].inputs[moveNum](row, col + Board::cols * channel), sizeof(float));
								}

					}
						
			}
				
			
			
		}
	}
	//std::cout << games[0].inputs[0];
	for (int gameNum = 0; gameNum < numGames; gameNum++)
	{
		for (int moveNum = 0; moveNum < games[gameNum].inputs.size(); moveNum++)
		{
			posNum++;
			// output << ((game.result == COLOR_YELLOW) ? "Y " : (game.result == COLOR_DRAW ? "D " : "R "));
			//std::cout << results[gameNum] << "\n";
			//std::cout << inputs[gameNum * Board::cols * Board::rows+ moveNum] << "\n\n";
			output.write((char*)&results[gameNum], sizeof(float));
		}
	}
	std::cout << numPositions << "\n";
	std::cout << posNum << "\n";
	output.close();
	delete[] results;
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
				move = go(pos, 0, false, 0);
				pos.doMove(move);
			}
		}
		if ((pos.gameResult() == COLOR_YELLOW && (game % 2)) || (pos.gameResult() == COLOR_RED && !(game % 2))) score-= 1.0;
		else if ((pos.gameResult() == COLOR_RED && (game % 2)) || (pos.gameResult() == COLOR_YELLOW && !(game % 2))) score += 1.0;
	}
	std::cout << "score: " << score << "\n";
	//exit(1);
}

void calculate()
{
	std::ifstream output("games.bin", std::ios::binary);
	int numPositions;
	output.read((char*)&numPositions, sizeof(int));
	for (int pos = 0; pos < 100; pos++)
	{
		Eigen::MatrixXf posInput(6, 7 * 12);
		for (int row = 0; row < Board::rows; row++)
		{
			for (int col = 0; col < Board::cols; col++)
			{
				for (int channel = 0; channel < 12; channel++)
				{
					output.read((char*)&posInput(row, col + Board::cols * channel), sizeof(float));
				}

			}
		}
		std::cout << pos << ":" << "\n";
		std::cout<<network.calculateOutput(posInput) << "\n";
	}
}

}