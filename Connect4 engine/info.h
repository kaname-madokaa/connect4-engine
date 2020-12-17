#pragma once

#ifndef EMBEDED
struct layerInfo
{
	//used when layer is dense
	const int layerSize = -1;
	//used when layer is conv
	const int filter_rows = -1;
	const int filter_cols = -1;
	const int input_rows = -1;
	const int input_cols = -1;
	const int input_channels = -1;
	const int output_channels = -1;
};



inline constexpr char layerTypes[] = { 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd' };
inline constexpr int layerNum = 8;
inline constexpr layerInfo layersInformation[] =
{
	//layer 1
	{
		//used when layer is dense
		 300 * 5 * 6,
		 //used when layer is conv
		  2, 2, 6, 7, 3, 300
	 },
	//layer 2
	{
		//used when layer is dense
		 300 * 4 * 5,
		 //used when layer is conv
		  2, 2, 5, 6, 300, 300
	 },
	//layer 3
	{
		//used when layer is dense
		 300 * 3 * 4,
		 //used when layer is conv
		 2, 2, 4, 5, 300, 300
	},
	//layer 4
	{
		//used when layer is dense
		 300 * 2 * 3,
		 //used when layer is conv
		  2, 2, 3, 4, 300, 300
	},
	//layer 5
	{
		//used when layer is dense
		 300 * 1 * 2,
		 //used when layer is conv
		  2, 2, 2, 3, 300, 300
	},
	//layer 6
	{
		//used when layer is dense
		 300 * 1 * 2,
		 //used when layer is conv
		  1, 1, 1, 2, 300, 300
	},
	//layer 7
	{
		//used when layer is dense
		 600,
		 //used when layer is conv
		 -1, -1, -1, -1, -1, -1
	},
	//layer 8
	{
		//used when layer is dense
		 1,
		 //used when layer is conv
		  -1, -1, -1, -1, -1, -1
	 },


};

inline constexpr int input_rows = 6;
inline constexpr int input_cols = 7;
inline constexpr int input_channels = 3; 
#else 
struct layerInfo
{
	//used when layer is dense
	 int layerSize = -1;
	//used when layer is conv
	 int filter_rows = -1;
	 int filter_cols = -1;
	 int input_rows = -1;
	 int input_cols = -1;
	 int input_channels = -1;
	 int output_channels = -1;
};


#endif