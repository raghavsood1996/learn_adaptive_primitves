#pragma once

#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include "fssimplewindow.h"
#include "Bitmap.h"
#include "motion_prim.h"
#include <queue>
#include <unordered_set>
#include <math.h>
#include <torch/script.h>
#if !defined(MAX)
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif
#define NUMOFDIRS 8

int temp = 0;

struct plan_stats
{
	double planning_time;
	double plan_cost;
	int exp_per_sec;
};


// Transforms the original map for inflating the obstacles by width of the car
void map_transform(CharBitmap *map)
{
	int inflt_coeff = 4;
	int itr;
	for (int i = 0; i < 400; i++)
	{
		for (int j = 0; j < 400; j++)
		{
			itr = 0;
			if (!map->getPixel(i, j))
			{
				for (int x = i - inflt_coeff; x <= i + inflt_coeff; x++)
				{
					for (int y = j - inflt_coeff; y <= j + inflt_coeff; y++)
					{
						itr++;
						if (x < 400 && x >= 0 && y >= 0 && y < 400)
						{
							map->setTransPixel(x, y, '4');
						}
					}
				}
			}
		}
	}
}

//Priority function for heuristic planner
class priority2
{
public:
	int operator()(const state2d &s1, const state2d &s2)
	{
		return s1.gval > s2.gval;
	}
};

//Planner that calculates heuristics
void heuristic_planner(state2d **states, CharBitmap *map, node start)
{

	int dX[NUMOFDIRS] = {-1, -1, -1, 0, 0, 1, 1, 1};
	int dY[NUMOFDIRS] = {-1, 0, 1, -1, 1, -1, 0, 1};

	for (int i = 0; i < 400; i++)
	{
		for (int j = 0; j < 400; j++)
		{
			states[i][j].x = i;
			states[i][j].y = j;
			states[i][j].gval = INT_MAX; //intializing the gvalues to a very large number
			states[i][j].isclosed = false;
		}
	}


	
	priority_queue<state2d, vector<state2d>, priority2> openlist;
	states[start.x][start.y].gval = 0;
	openlist.push(states[start.x][start.y]);
	state2d current_state;
	int iter = 0;
	clock_t beginTime;
	beginTime = clock();
	while (!openlist.empty())
	{
		iter++;
		//cout << iter <<"iter"<< endl;
		current_state = openlist.top();
		openlist.pop();
		states[current_state.x][current_state.y].isclosed = true; //putting state in closed list

		for (int dir = 0; dir < NUMOFDIRS; dir++)
		{
			int newx = current_state.x + dX[dir];
			int newy = current_state.y + dY[dir];

			if (newx >= 1 && newx < 400 && newy >= 1 && newy < 400)
			{

				if (map->isTransFree(newx, newy) && states[newx][newy].isclosed == false)
				{ //if free and not closed
					//cout << newx << " x " << newy << " y " << endl;
					if (states[newx][newy].gval > current_state.gval + sqrt(dX[dir] * dX[dir] + dY[dir] * dY[dir]))
					{
						states[newx][newy].gval = current_state.gval + sqrt(dX[dir] * dX[dir] + dY[dir] * dY[dir]);
						openlist.push(states[newx][newy]);
					}
				}
			}
		}
	}
	
	float time_passed = (float(clock() - beginTime)) / CLOCKS_PER_SEC;
	//cout << "Time Used: " << time_passed << " seconds." << endl;
};

//priority function for open list of main planner
class priority
{
public:
	float eps = 2;
	int operator()(const node &s1, const node &s2)
	{
		return s1.gval + eps * s1.hval > s2.gval + eps * s2.hval;
	}
};

//just a test heuristics that only takes eucledian dist
float heuristic(int start_x, int start_y, int goal_x, int goal_y)
{
	return sqrt(2) * MIN(abs(start_x - goal_x), abs(start_y - goal_y)) + MAX(abs(start_x - goal_x), abs(start_y - goal_y)) - MIN(abs(start_x - goal_x), abs(start_y - goal_y));
}

// util function for unordered set of nodes
struct nodehasher
{
	size_t operator()(const node &thenode) const
	{
		return hash<string>{}(thenode.toString());
	}
};

// util function for unordred set of nodes
struct nodeComparator
{
	bool operator()(const node &lhs, const node &rhs) const
	{
		return lhs == rhs;
	}
};

// Main Planner
vector<node> planner( state2d **states, CharBitmap *env, lattice_graph *graph, node start, node goal, plan_stats *stats, unordered_map<config, bool, confighasher, configComparator> &reed_map, KDtree<float, SAMPLES, 3> &kdtree,
							  array<array<float, 3>, SAMPLES> &tree_data, bool use_net)
{
	clock_t beginTime;
	beginTime = clock();
	vector<node> plan;
	priority_queue<node, vector<node>, priority> openlist;
	unordered_set<node, nodehasher, nodeComparator> closed_list;
	start.gval = 0;
	openlist.push(start);
	node lastnode;
	int itr = 0;
	float plan_time;
	while (!openlist.empty())
	{
		itr++;
		node *curr_node = new node;
		*curr_node = openlist.top();
		openlist.pop();
		closed_list.insert(*curr_node);
		if (curr_node->x == goal.x && curr_node->y == goal.y && curr_node->theta == goal.theta)
		{ //check expansion of goal
			lastnode = *curr_node;
			break;
		}
		for (node successor : graph->getsuccessor(*curr_node, env, states, goal, reed_map,kdtree,tree_data,use_net))
		{

			if (closed_list.find(successor) == closed_list.end() && successor.x > 0 && successor.x < 400 && successor.y > 10 && successor.y < 400)
			{

				if (successor.gval > curr_node->gval + successor.pre_cost)
				{
					successor.gval = curr_node->gval + successor.pre_cost;
					successor.hval = states[successor.x][successor.y].gval; //heuristic from backward search
					successor.parent = curr_node;
					openlist.push(successor);
				}
			}
		}

		//if planner is taking more than 20 seonds to give out the plan just stay there so that other vehicles can pass and you can replan again
		plan_time = (float(clock() - beginTime)) / CLOCKS_PER_SEC;
	}

	if(use_net){
		cout<<"Stats with Net"<<endl;
	}

	else{
		cout<<"Stats without Net"<<endl;
	}

	float time_passed = (float(clock() - beginTime)) / CLOCKS_PER_SEC;
	cout << "Time Used: " << time_passed << " seconds." << endl;
	stats->planning_time = time_passed;
	cout << "Total Expansions: " << itr << endl;
	cout << "Expansions per second: " << itr / time_passed << "" << endl;
	stats->exp_per_sec = itr / time_passed;

	float cost_sum = 0;
	if (openlist.empty())
	{
		cout << "No Solution to the problem exist" << endl;
		node temp;
		
		stats->exp_per_sec = 0;
		plan.insert(plan.begin(), temp);
	}

	else
	{

		node *itr = lastnode.parent;
		cost_sum += lastnode.pre_cost;
		plan.insert(plan.begin(), lastnode);

		while (itr != NULL)
		{
			node temp_node = *itr;
			cost_sum += temp_node.pre_cost;
			plan.insert(plan.begin(), temp_node);
			itr = itr->parent;
		}
	}
	cout << "Path Cost: " << cost_sum << endl;
	cout<<" "<<endl;
	stats->plan_cost = cost_sum;
	return plan;
}