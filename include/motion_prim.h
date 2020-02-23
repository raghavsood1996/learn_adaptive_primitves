#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <torch/torch.h>
#include "Bitmap.h"
#include "reeds_shepp.h"
#include "col_model.h"
#include "KDtree.hpp"

#define INT_MAX 100000
using namespace std;



struct confighasher
{
	size_t operator()(const config &thenode) const
	{
		return hash<string>{}(thenode.toString());
	}
};

// util function for unordred set of nodes
struct configComparator
{
	bool operator()(const config &lhs, const config &rhs) const
	{
		return lhs == rhs;
	}
};


// for 2d heuristics search on a x and y grid space
struct state2d
{
	int x;
	int y;
	bool isclosed;
	float gval;
	state2d()
	{
		this->gval = INT_MAX;
	}
};

// node describing a state in the lattice graph
struct node
{
	int x;
	int y;
	int theta;
	float gval;
	float hval;
	float pre_cost;
	node *parent;
	node()
	{
		this->gval = INT_MAX;
		this->parent = NULL;
	}

	node(int x, int y, int theta)
	{
		this->x = x;
		this->y = y;
		this->theta = theta;
		this->parent = NULL;
		this->pre_cost = 0;
		this->gval = INT_MAX;
	}

	bool operator==(const node &rhs) const
	{

		if (this->x != rhs.x)
		{
			return false;
		}
		if (this->y != rhs.y)
		{
			return false;
		}
		if (this->theta != rhs.theta)
		{
			return false;
		}
		return true;
	}

	friend ostream &operator<<(ostream &os, const node &w)
	{
		os << "***** Node *****" << endl
		   << endl;
		os << "X: ";
		os << w.x;
		os << endl;
		os << "Y: ";
		os << w.y;
		os << endl;
		os << "Heading: ";
		os << w.theta;
		os << endl;
		os << "G value: ";
		os << w.gval;
		os << endl;
		os << "H value: ";
		os << w.hval;
		os << endl;
		os << "Pre cost: ";
		os << w.pre_cost;
		os << endl;
		return os;
	}

	string toString() const
	{
		string temp = "";

		temp += to_string(this->x) + ",";
		temp += to_string(this->y) + ",";
		temp += to_string(this->theta) + ",";
		temp = temp.substr(0, temp.length() - 1);

		return temp;
	}
};

struct motion_prim
{
	int dx;
	int dy;
	int dtheta;
	float cost;
};

float t_area(int x1, int y1, int x2, int y2,
			 int x3, int y3)
{
	return abs((x1 * (y2 - y3) + x2 * (y3 - y1) +
				x3 * (y1 - y2)) /
			   2.0);
}

bool check_obst(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, int x, int y)
{
	//cout << "point being checked is" << x << " " << y << endl;
	/* Calculate area of rectangle ABCD */
	float A = t_area(x1, y1, x2, y2, x3, y3) +
			  t_area(x1, y1, x4, y4, x3, y3);

	/* Calculate area of triangle PAB */
	float A1 = t_area(x, y, x1, y1, x2, y2);

	/* Calculate area of triangle PBC */
	float A2 = t_area(x, y, x2, y2, x3, y3);

	/* Calculate area of triangle PCD */
	float A3 = t_area(x, y, x3, y3, x4, y4);

	/* Calculate area of triangle PAD */
	float A4 = t_area(x, y, x1, y1, x4, y4);

	/* Check if sum of A1, A2, A3 and A4  
       is same as A */
	//cout << "A" << A << endl;
	//cout << "triangles_combined" << A1 + A2 + A3 + A4 << endl;
	return (A == A1 + A2 + A3 + A4);
}

bool isValid(CharBitmap *env, config state)
{

	int width = 4;
	int length = 8;
	double angle = -state.theta * 3.14 / 180;

	//coordinates of the quadrilateral
	int x1 = state.x + (length) * cos(angle) - (width) * sin(angle);
	int y1 = state.y + (length) * sin(angle) + (width) * cos(angle);

	int x2 = state.x + (-length) * cos(angle) - (width) * sin(angle);
	int y2 = state.y + (-length) * sin(angle) + (width) * cos(angle);

	int x3 = state.x + (-length) * cos(angle) - (-width) * sin(angle);
	int y3 = state.y + (-length) * sin(angle) + (-width) * cos(angle);

	int x4 = state.x + (length) * cos(angle) - (-width) * sin(angle);
	int y4 = state.y + (length) * sin(angle) + (-width) * cos(angle);

	//cout << x1 << " " << y1 << " " << x2 << " " << y2 << " " << x3 << " " << y3 << " " << x4 << " " << y4 << endl;
	//cout << start_x << " " << start_y << endl;

	for (int start_x = state.x - 10; start_x <= state.x + 10; start_x++)
	{

		for (int start_y = state.y - 10; start_y <= state.y + 10; start_y++)
		{

			if (!env->isFree(start_x, start_y) /*&& check_point(x1,y1,x2,y2,x3,y3,x4,y4,start_x,start_y)*/)
			{
				//cout << "not free point found..now checking if its inside the car" << endl;
				if (check_obst(x1, y1, x2, y2, x3, y3, x4, y4, start_x, start_y))
				{
					//cout << "invalid configuration demanded" << endl;
					return 0;
				}
			}
		}
	}
	//cout << "valid_config" << endl;
	return 1;
}

double distance(double x1, double y1, double x2, double y2)
{

	return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

//callback function for reedshepp path
static int reeds_cb(vector<config> *path_samples, double q[3], void *user_data)
{
		config temp(q[0], q[1], q[2]);
		path_samples->push_back(temp);
		return 0;
}

// collision check function for either dubuns or reed shepps path
bool dubins_collision_check(vector<config> &path_samples, CharBitmap *map)
	{
		for (config point : path_samples)
		{
			if (!map->isTransFree(point.x, point.y))
			{
				return 0;
			}
			if (!isValid(map, point))
			{

				return 0;
			}
		}
		//cout << "valid_reeds" << endl;
		return 1;
	}




class lattice_graph
{
private:
	vector<node> the_graph;
	motion_prim fwd;
	motion_prim back;
	motion_prim ccw;
	motion_prim cw;
	motion_prim fws;
	vector<motion_prim> prim_0;				//storing primitives for angle 0 degrees
	vector<motion_prim> prim_45;			//storing primitives for angle 45 degrees
	vector<motion_prim> prim_90;			//storing primitives for angle 90 degrees
	vector<motion_prim> prim_135;			//storing primitives for angle 135 degrees
	vector<motion_prim> prim_180;			//storing primitives for angle 180 degrees
	vector<motion_prim> prim_225;			//storing primitives for angle 225 degrees
	vector<motion_prim> prim_270;			//storing primitives for angle 270 degrees
	vector<motion_prim> prim_315;			//storing primitives for angle 315 degrees
	vector<motion_prim> prim_360;			//storing primitives for angle 360 degrees
	map<int, vector<motion_prim>> prim_map; //mapping angles to motion primitives
	int back_cost = 5;						//cost for backward motion primitive
	int cw_cost = 18;						//cost for 1/16 counter-clockwise motion primitive
	int ccw_cost = 18;						//cost for 1/16 clockwise motion primitive
	int fwd_cost = 1;						//cost for 1 step forward
	int fwd_step_cost = 8;					//cost for moving more than one step forward
	int fwd_step = 8;						//size for forward step motion primitive
	float diag_mult = 1.42;

public:
	// sets motion primitives for the graph
	void set_motion_prims()
	{
		//motion_prims for 0 degrees heading
		prim_0.push_back(fwd);
		prim_0.push_back(back);
		prim_0.push_back(ccw);
		prim_0.push_back(cw);
		prim_0.push_back(fws);
		prim_0[0].dx = 1; //forward one step
		prim_0[0].dy = 0;
		prim_0[0].dtheta = 0;
		prim_0[0].cost = fwd_cost;

		prim_0[1].dx = 15; //ccw 1/16 turn
		prim_0[1].dy = 6;
		prim_0[1].dtheta = 1;
		prim_0[1].cost = ccw_cost;

		prim_0[2].dx = 15; //cw 1/16 turn
		prim_0[2].dy = -6;
		prim_0[2].dtheta = -1;
		prim_0[2].cost = ccw_cost;

		prim_0[3].dx = -1; //backward one step
		prim_0[3].dy = 0;
		prim_0[3].dtheta = 0;
		prim_0[3].cost = back_cost;

		prim_0[4].dx = fwd_step; //forward some step
		prim_0[4].dy = 0;
		prim_0[4].dtheta = 0;
		prim_0[4].cost = fwd_step_cost;

		//motion_prims for 45 degrees heading

		prim_45.push_back(fwd);
		prim_45.push_back(back);
		prim_45.push_back(ccw);
		prim_45.push_back(cw);
		prim_45.push_back(fws);

		prim_45[0].dx = 1; //forward one step
		prim_45[0].dy = 1;
		prim_45[0].dtheta = 0;
		prim_45[0].cost = fwd_cost * diag_mult;

		prim_45[1].dx = 6; //ccw 1/16 turn
		prim_45[1].dy = 15;
		prim_45[1].dtheta = 1;
		prim_45[1].cost = ccw_cost;

		prim_45[2].dx = 15; //cw 1/16 turn
		prim_45[2].dy = 6;
		prim_45[2].dtheta = -1;
		prim_45[2].cost = ccw_cost;

		prim_45[3].dx = -1; //backward one step
		prim_45[3].dy = -1;
		prim_45[3].dtheta = 0;
		prim_45[3].cost = back_cost * diag_mult;

		prim_45[4].dx = fwd_step; //forward one step
		prim_45[4].dy = fwd_step;
		prim_45[4].dtheta = 0;
		prim_45[4].cost = fwd_step_cost * diag_mult;

		//motion_prims for 90 degrees heading
		prim_90.push_back(fwd);
		prim_90.push_back(back);
		prim_90.push_back(ccw);
		prim_90.push_back(cw);
		prim_90.push_back(fws);

		prim_90[0].dx = 0; //forward one step
		prim_90[0].dy = 1;
		prim_90[0].dtheta = 0;
		prim_90[0].cost = fwd_cost;

		prim_90[1].dx = -6; //ccw 1/16 turn
		prim_90[1].dy = 15;
		prim_90[1].dtheta = 1;
		prim_90[1].cost = ccw_cost;

		prim_90[2].dx = 6; //cw 1/16 turn
		prim_90[2].dy = 15;
		prim_90[2].dtheta = -1;
		prim_90[2].cost = ccw_cost;

		prim_90[3].dx = 0; //backward one step
		prim_90[3].dy = -1;
		prim_90[3].dtheta = 0;
		prim_90[3].cost = back_cost;

		prim_90[4].dx = 0; //forward one step
		prim_90[4].dy = fwd_step;
		prim_90[4].dtheta = 0;
		prim_90[4].cost = fwd_step_cost;

		//motion_prims for 135 degrees heading
		prim_135.push_back(fwd);
		prim_135.push_back(back);
		prim_135.push_back(ccw);
		prim_135.push_back(cw);
		prim_135.push_back(fws);

		prim_135[0].dx = -1; //forward one step
		prim_135[0].dy = 1;
		prim_135[0].dtheta = 0;
		prim_135[0].cost = fwd_cost * diag_mult;

		prim_135[1].dx = -15; //ccw 1/16 turn
		prim_135[1].dy = 6;   //
		prim_135[1].dtheta = 1;
		prim_135[1].cost = ccw_cost;

		prim_135[2].dx = -6; //cw 1/16 turn
		prim_135[2].dy = 15;
		prim_135[2].dtheta = -1;
		prim_135[2].cost = ccw_cost;

		prim_135[3].dx = 1; //backward one step
		prim_135[3].dy = -1;
		prim_135[3].dtheta = 0;
		prim_135[3].cost = back_cost * diag_mult;

		prim_135[0].dx = -fwd_step; //forward steps
		prim_135[0].dy = fwd_step;
		prim_135[0].dtheta = 0;
		prim_135[0].cost = fwd_step_cost * diag_mult;

		//motion_prims for 180 degrees heading
		prim_180.push_back(fwd);
		prim_180.push_back(back);
		prim_180.push_back(ccw);
		prim_180.push_back(cw);
		prim_180.push_back(fws);

		prim_180[0].dx = -1; //forward one step
		prim_180[0].dy = 0;
		prim_180[0].dtheta = 0;
		prim_180[0].cost = fwd_cost;

		prim_180[1].dx = -15; //ccw 1/16 turn
		prim_180[1].dy = -6;
		prim_180[1].dtheta = 1;
		prim_180[1].cost = ccw_cost;

		prim_180[2].dx = -15; //cw 1/16 turn
		prim_180[2].dy = 6;
		prim_180[2].dtheta = -1;
		prim_180[2].cost = ccw_cost;

		prim_180[3].dx = 1; //backward one step
		prim_180[3].dy = 0;
		prim_180[3].dtheta = 0;
		prim_180[3].cost = back_cost;

		prim_180[4].dx = -fwd_step; //forward step
		prim_180[4].dy = 0;
		prim_180[4].dtheta = 0;
		prim_180[4].cost = fwd_step_cost;

		//motion_prims for 225 degrees heading
		prim_225.push_back(fwd);
		prim_225.push_back(back);
		prim_225.push_back(ccw);
		prim_225.push_back(cw);
		prim_225.push_back(fws);

		prim_225[0].dx = -1; //forward one step
		prim_225[0].dy = -1;
		prim_225[0].dtheta = 0;
		prim_225[0].cost = fwd_cost * diag_mult;

		prim_225[1].dx = -6;  //ccw 1/16 turn
		prim_225[1].dy = -15; //
		prim_225[1].dtheta = 1;
		prim_225[1].cost = ccw_cost;

		prim_225[2].dx = -15; //cw 1/16 turn
		prim_225[2].dy = -6;
		prim_225[2].dtheta = -1;
		prim_225[2].cost = ccw_cost;

		prim_225[3].dx = 1; //backward one step
		prim_225[3].dy = 1;
		prim_225[3].dtheta = 0;
		prim_225[3].cost = back_cost * diag_mult;

		prim_225[4].dx = -fwd_step; //forward one step
		prim_225[4].dy = -fwd_step;
		prim_225[4].dtheta = 0;
		prim_225[4].cost = fwd_step_cost * diag_mult;

		//motion_prims for 270 degrees heading
		prim_270.push_back(fwd);
		prim_270.push_back(back);
		prim_270.push_back(ccw);
		prim_270.push_back(cw);
		prim_270.push_back(fws);

		prim_270[0].dx = 0; //forward one step
		prim_270[0].dy = -1;
		prim_270[0].dtheta = 0;
		prim_270[0].cost = fwd_cost;

		prim_270[1].dx = 6; //ccw 1/16 turn
		prim_270[1].dy = -15;
		prim_270[1].dtheta = 1;
		prim_270[1].cost = ccw_cost;

		prim_270[2].dx = -6; //cw 1/16 turn
		prim_270[2].dy = -15;
		prim_270[2].dtheta = -1;
		prim_270[2].cost = ccw_cost;

		prim_270[3].dx = 0; //backward one step
		prim_270[3].dy = 1;
		prim_270[3].dtheta = 0;
		prim_270[3].cost = back_cost;

		prim_270[4].dx = 0; //forward step
		prim_270[4].dy = -fwd_step;
		prim_270[4].dtheta = 0;
		prim_270[4].cost = fwd_step_cost;

		//motion_prims for 315 degrees heading
		prim_315.push_back(fwd);
		prim_315.push_back(back);
		prim_315.push_back(ccw);
		prim_315.push_back(cw);
		prim_315.push_back(fws);

		prim_315[0].dx = 1; //forward one step
		prim_315[0].dy = -1;
		prim_315[0].dtheta = 0;
		prim_315[0].cost = fwd_cost * diag_mult;

		prim_315[1].dx = 15; //ccw 1/16 turn
		prim_315[1].dy = -6;
		prim_315[1].dtheta = 1;
		prim_315[1].cost = ccw_cost;

		prim_315[2].dx = 6; //cw 1/16 turn
		prim_315[2].dy = -15;
		prim_315[2].dtheta = -1;
		prim_315[2].cost = ccw_cost;

		prim_315[3].dx = -1; //backward one step
		prim_315[3].dy = 1;
		prim_315[3].dtheta = 0;
		prim_315[3].cost = back_cost * diag_mult;

		prim_315[4].dx = fwd_step; //forward one step
		prim_315[4].dy = -fwd_step;
		prim_315[4].dtheta = 0;
		prim_315[4].cost = fwd_step_cost * diag_mult;

		prim_map[0] = prim_0;
		prim_map[45] = prim_45;
		prim_map[90] = prim_90;
		prim_map[135] = prim_135;
		prim_map[180] = prim_180;
		prim_map[225] = prim_225;
		prim_map[270] = prim_270;
		prim_map[315] = prim_315;
	}

	static int dubins_cb1(vector<config> *path_samples, double q[3], double x, void *user_data)
	{
		config temp(q[0], q[1], q[2]);

		path_samples->push_back(temp);
		return 0;
	}

	
	// generates successors in the lattice graph

	//New update to add Reeds path as a dynamic motion primitive
	//changes made are adding the 2d heuristic values as argument to the function and goal to the arguments
	vector<node> getsuccessor(node curr_node, CharBitmap *map, state2d **states, node goal_node, unordered_map<config, bool, confighasher, configComparator> &reed_map, KDtree<float, SAMPLES, 3> &kdtree,
							  array<array<float, 3>, SAMPLES> &tree_data, bool use_net)
	{
		vector<node> succesors;
		node succesor;
		int currx = curr_node.x;
		int curry = curr_node.y;
		int curr_heading = curr_node.theta;

		if (curr_heading >= 360)
		{
			curr_heading = curr_heading - 360;
		}

		for (int dir = 0; dir <= 4; dir++)
		{

			if (!collision_check(currx, curry, map, dir, curr_heading))
			{

				continue;
			}
			else
			{

				succesor.x = currx + prim_map[curr_heading][dir].dx;
				succesor.y = curry + prim_map[curr_heading][dir].dy;
				succesor.theta = curr_heading + prim_map[curr_heading][dir].dtheta * 45;
				if (succesor.theta < 0)
				{
					succesor.theta += 360;
				}
				succesor.pre_cost = prim_map[curr_heading][dir].cost;
				succesors.push_back(succesor);
			}

			//activate Reeds Shepp path when you are a certain distance away from the goal
            
			if (!use_net && dir == 4)
			{
				//clock_t begin = clock();

				double curr[] = {currx, curry, curr_heading * 3.142 / 180.0};
				double goal[] = {goal_node.x, goal_node.y, goal_node.theta * 3.142 / 180.0};
				double turning_radius = 22.0;
				ReedsSheppStateSpace space(turning_radius);
				vector<config> path_samples;
				ReedsSheppStateSpace::ReedsSheppPath path;
				path = space.reedsShepp(curr, goal);
				space.sample(&path_samples, curr, goal, 2, reeds_cb, NULL);

				if (dubins_collision_check(path_samples, map))
				{
					//out << "dynamic prim time" << (float(clock() - begin)) / CLOCKS_PER_SEC << endl;

					goal_node.pre_cost = int(space.distance(curr, goal));
					//cout<<"reed sheep length "<<space.distance(curr,goal)<<endl;
					succesors.push_back(goal_node);
				}
			}

			else if(use_net && dir == 4){

				std::array<float, 3> curr = {currx, curry, curr_heading * 3.142 / 180.0};

				auto near_idx = kdtree.find_k_nearest<Distance::euclidean>(1, curr);

				auto nearest = tree_data[near_idx[0]];

				

				config temp(nearest[0], nearest[1], int(nearest[2] * 180 / 3.14));

				if (reed_map[temp])
				{
				
					double curr[] = {currx, curry, curr_heading * 3.142 / 180.0};
					double goal[] = {goal_node.x, goal_node.y, goal_node.theta * 3.142 / 180.0};
					double turning_radius = 22.0;
					ReedsSheppStateSpace space(turning_radius);
					vector<config> path_samples;
					ReedsSheppStateSpace::ReedsSheppPath path;
					path = space.reedsShepp(curr, goal);
					space.sample(&path_samples, curr, goal, 3, reeds_cb, NULL);

					if(dubins_collision_check(path_samples,map)){

						goal_node.pre_cost = int(space.distance(curr, goal));
						succesors.push_back(goal_node);

					}
					
				}
			}

			
		}
		return succesors;
	}

	//Actual collision check for Reeds Path
	

	// collision checking for all non-dynamic motion primitives
	bool collision_check(int currx, int curry, CharBitmap *map, int dir, int heading)
	{

		//for heading 0 forward
		if (dir == 0 && heading == 0)
		{
			for (int i = currx + 8; i <= currx + 9; i++)
			{
				for (int j = curry - 4; j <= curry + 4; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision if map is not free
					}
				}
			}
			return 1;
		}

		//for heading 0 ccw turn
		else if (dir == 1 && heading == 0)
		{
			for (int i = currx + 8; i <= currx + 12; i++)
			{
				for (int j = curry - 4; j <= curry + 4; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			int x = currx + 12;
			int y_start = curry - 4;
			int y_end = curry + 9;

			for (x; x - currx <= 18; x++)
			{

				for (int y = y_start; y <= y_end; y++)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				y_start++;
				y_end++;
			}

			int x_new = currx + 19;
			y_start = curry + 3;
			y_end = curry + 14;

			for (x_new; x_new - currx <= 24; x_new++)
			{

				for (int y = y_start; y <= y_end; y++)
				{
					if (!map->isTransFree(x_new, y))
					{
						return 0; //collision
					}
				}
				y_start++;
				y_end--;
			}
			return 1;
		}

		//for heading 0 and cw turn
		else if (dir == 2 && heading == 0)
		{
			for (int i = currx + 8; i <= currx + 12; i++)
			{
				for (int j = curry - 4; j <= curry + 4; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}

			int x = currx + 12;
			int y_start = curry + 4;
			int y_end = curry - 9;
			for (x; x - currx <= 18; x++)
			{

				for (int y = y_start; y >= y_end; y--)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				y_start--;
				y_end--;
			}

			int x_new = currx + 19;
			y_start = curry - 3;
			y_end = curry - 14;

			for (x_new; x_new - currx <= 24; x_new++)
			{

				for (int y = y_start; y >= y_end; y--)
				{
					if (!map->isTransFree(x_new, y))
					{
						return 0; //collision
					}
				}

				y_start--;
				y_end++;
			}
			return 1;
		}

		//for heading 0 and backward
		else if (dir == 3 && heading == 0)
		{
			for (int i = currx - 8; i >= currx - 9; i--)
			{
				for (int j = curry - 4; j <= curry + 4; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		//for heading 0 and few steps forward
		else if (dir == 4 && heading == 0)
		{
			for (int i = currx + 8; i <= currx + 1 + fwd_step; i++)
			{
				for (int j = curry - 4; j <= curry + 4; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision if map is not free
					}
				}
			}
			return 1;
		}

		//for heading 45 and forward
		else if (dir == 0 && heading == 45)
		{
			int j = curry + 10;
			for (int i = currx + 4; i <= currx + 10; i++)
			{
				if (!map->isTransFree(i, j))
				{
					return 0; //collision
				}
				j--;
			}
			return 1;
		}

		//for heading 45 and ccw turn
		else if (dir == 1 && heading == 45)
		{
			for (int i = currx + 2; i <= currx + 10; i++)
			{
				for (int j = curry + 4; j <= curry + 23; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		// for heading 45 and cw turn
		else if (dir == 2 && heading == 45)
		{
			for (int i = currx + 3; i <= currx + 23; i++)
			{
				for (int j = curry + 2; j <= curry + 10; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		//for heading 45 and backwards
		else if (dir == 3 && heading == 45)
		{
			int j = curry - 10;
			for (int i = currx - 4; i <= currx + 2; i++)
			{
				if (!map->isTransFree(i, j))
				{
					return 0; //collision
				}
				j--;
			}
			return 1;
		}

		//for heading 45 and few step forward
		else if (dir == 4 && heading == 45)
		{
			for (currx; currx <= currx + fwd_step; currx++)
			{
				int j = curry + 10;
				for (int i = currx + 4; i <= currx + 10; i++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
					j--;
				}
				curry++;
			}
			return 1;
		}

		//for heading 90 and forward
		else if (dir == 0 && heading == 90)
		{
			for (int i = curry + 8; i <= curry + 9; i++)
			{
				for (int j = currx - 4; j <= currx + 4; j++)
				{
					if (!map->isTransFree(j, i))
					{
						return 0; //collision if map is not free
					}
				}
			}
			return 1;
		}

		//for heading 90 and ccw turn
		else if (dir == 1 && heading == 90)
		{
			for (int i = curry + 8; i <= curry + 12; i++)
			{
				for (int j = currx - 4; j <= currx + 4; j++)
				{
					if (!map->isTransFree(j, i))
					{
						return 0; //collision
					}
				}
			}
			int y = curry + 13;
			int x_start = currx + 4;
			int x_end = currx - 9;
			for (y; y - curry <= 18; y++)
			{

				for (int x = x_start; x > x_end; x--)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				x_start--;
				x_end--;
			}

			int y_new = curry + 19;
			x_start = currx - 3;
			x_end = currx - 15;

			for (y_new; y_new - curry <= 24; y_new++)
			{

				for (int x = x_start; x > x_end; x--)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				x_start--;
				x_end++;
			}
			return 1;
		}

		//for heading 90 and cw turn
		else if (dir == 2 && heading == 90)
		{

			for (int i = curry + 8; i <= curry + 12; i++)
			{
				for (int j = currx - 4; j <= currx + 4; j++)
				{
					if (!map->isTransFree(j, i))
					{
						return 0; //collision
					}
				}
			}
			int y = curry + 13;
			int x_start = currx - 4;
			int x_end = currx + 9;
			for (y; y - curry <= 18; y++)
			{

				for (int x = x_start; y < x_end; x++)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				x_start++;
				x_end++;
			}

			int y_new = curry + 19;
			x_start = currx + 3;
			x_end = currx + 15;

			for (y_new; y_new - curry <= 24; y_new++)
			{

				for (int x = x_start; x < x_end; x++)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				x_start++;
				x_end--;
			}
			return 1;
		}

		//for heading 90 and backwards
		else if (dir == 3 && heading == 90)
		{
			for (int i = curry - 8; i <= curry - 9; i++)
			{
				for (int j = currx - 4; j <= currx + 4; j++)
				{
					if (!map->isTransFree(j, i))
					{
						return 0; //collision if map is not free
					}
				}
			}
			return 1;
		}

		//for heading 90 and few steps forward
		else if (dir == 4 && heading == 90)
		{
			for (int i = curry + 8; i <= curry + fwd_step + 9; i++)
			{
				for (int j = currx - 4; j <= currx + 4; j++)
				{
					if (!map->isTransFree(j, i))
					{
						return 0; //collision if map is not free
					}
				}
			}
			return 1;
		}

		//for heading 135 and forward
		else if (dir == 0 && heading == 135)
		{
			int j = curry + 10;
			for (int i = currx - 4; i >= currx - 10; i--)
			{
				if (!map->isTransFree(i, j))
				{
					return 0; //collision
				}
				j--;
			}
			return 1;
		}

		//for heading 135 and ccw turn
		else if (dir == 1 && heading == 135)
		{
			for (int i = currx - 3; i >= currx - 23; i--)
			{
				for (int j = curry + 2; j <= curry + 10; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		//for heading 135 and cw turn
		else if (dir == 2 && heading == 135)
		{
			for (int i = currx - 2; i >= currx - 10; i--)
			{
				for (int j = curry + 4; j <= curry + 23; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		//for heading 135 and backwards
		else if (dir == 3 && heading == 135)
		{
			int j = curry - 10;
			for (int i = currx + 4; i <= currx + 10; i++)
			{
				if (!map->isTransFree(i, j))
				{
					return 0; //collision
				}
				j++;
			}
			return 1;
		}

		//for heading 135 and few steps forward
		else if (dir == 4 && heading == 135)
		{

			for (currx; currx >= currx - fwd_step; currx--)
			{
				int j = curry + 10;
				for (int i = currx - 4; i >= currx - 10; i--)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
					j--;
				}
				curry++;
			}
			return 1;
		}

		//for heading 180 and forward
		else if (dir == 0 && heading == 180)
		{
			for (int i = currx - 8; i >= currx - 9; i--)
			{
				for (int j = curry - 4; j <= curry + 4; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		//for heading 180 and ccw turn
		else if (dir == 1 && heading == 180)
		{
			for (int i = currx - 8; i >= currx - 12; i--)
			{
				for (int j = curry - 4; j <= curry + 4; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			int x = currx - 13;
			int y_start = curry + 4;
			int y_end = curry - 9;
			for (x; x - currx >= -18; x--)
			{

				for (int y = y_start; y >= y_end; y--)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				y_start--;
				y_end--;
			}

			int x_new = currx - 19;
			y_start = curry - 3;
			y_end = curry - 15;

			for (x_new; x_new - currx >= -24; x_new--)
			{

				for (int y = y_start; y >= y_end; y--)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				y_start--;
				y_end++;
			}
			return 1;
		}

		//for heading 180 and cw turn
		else if (dir == 2 && heading == 180)
		{
			for (int i = currx - 8; i >= currx - 12; i--)
			{
				for (int j = curry - 4; j <= curry + 4; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			int x = currx - 13;
			int y_start = curry - 4;
			int y_end = curry + 9;
			for (x; x - currx >= -18; x--)
			{

				for (int y = y_start; y < y_end; y++)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				y_start++;
				y_end++;
			}

			int x_new = currx - 19;
			y_start = curry + 3;
			y_end = curry + 15;

			for (x_new; x_new - currx >= -24; x_new--)
			{

				for (int y = y_start; y <= y_end; y++)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				y_start++;

				y_end--;
			}
			return 1;
		}

		//for heading 180 and backwards
		else if (dir == 3 && heading == 180)
		{
			for (int i = currx + 8; i <= currx + 9; i++)
			{
				for (int j = curry - 4; j <= curry + 4; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision if map is not free
					}
				}
			}
			return 1;
		}

		//for heading 180 and few steps forward
		else if (dir == 4 && heading == 180)
		{
			for (int i = currx - 8; i >= currx - fwd_step - 9; i--)
			{
				for (int j = curry - 4; j <= curry + 4; j++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		//for heading 225 and forward
		else if (dir == 0 && heading == 225)
		{
			int j = curry - 10;
			for (int i = currx - 4; i <= currx + 2; i++)
			{
				if (!map->isTransFree(i, j))
				{
					return 0; //collision
				}
				j--;
			}
			return 1;
		}

		//for heading 225 and ccw turn
		else if (dir == 1 && heading == 225)
		{
			for (int i = currx - 2; i >= currx - 10; i--)
			{
				for (int j = curry - 4; j > curry - 23; j--)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		//for heading 225 and cw turn
		else if (dir == 2 && heading == 225)
		{
			for (int i = currx - 3; i >= currx - 23; i--)
			{
				for (int j = curry - 2; j >= curry - 10; j--)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		//for heading 225 and backwards
		else if (dir == 3 && heading == 225)
		{
			int j = curry + 10;
			for (int i = currx + 4; i <= currx + 10; i++)
			{
				if (!map->isTransFree(i, j))
				{
					return 0; //collision
				}
				j--;
			}
			return 1;
		}

		//for heading 225 and few steps forward
		else if (dir == 4 && heading == 225)
		{
			for (currx; currx >= currx - fwd_step; currx--)
			{
				int j = curry - 10;
				for (int i = currx - 4; i <= currx + 2; i++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
					j--;
				}
				curry--;
			}
			return 1;
		}

		//for heading 270 and forward
		else if (dir == 0 && heading == 270)
		{
			for (int i = curry - 8; i >= curry - 9; i--)
			{
				for (int j = currx - 4; j <= currx + 4; j++)
				{
					if (!map->isTransFree(j, i))
					{
						return 0; //collision if map is not free
					}
				}
			}
			return 1;
		}

		//for heading 270 and ccw turn
		else if (dir == 1 && heading == 270)
		{
			for (int i = curry - 8; i >= curry - 12; i--)
			{
				for (int j = currx - 4; j <= currx + 4; j++)
				{
					if (!map->isTransFree(j, i))
					{
						return 0; //collision
					}
				}
			}
			int y = curry - 13;
			int x_start = currx - 4;
			int x_end = currx + 9;
			for (y; y - curry >= -18; y--)
			{

				for (int x = x_start; y <= x_end; x++)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				x_start++;
				x_end++;
			}

			int y_new = curry - 19;
			x_start = currx + 3;
			x_end = currx + 15;

			for (y_new; y_new - curry >= -24; y_new--)
			{

				for (int x = x_start; x <= x_end; x++)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				x_start++;
				x_end--;
			}
			return 1;
		}

		//for heading 270 and cw turn
		else if (dir == 2 && heading == 270)
		{
			for (int i = curry - 8; i >= curry - 12; i--)
			{
				for (int j = currx - 4; j <= currx + 4; j++)
				{
					if (!map->isTransFree(j, i))
					{
						return 0; //collision
					}
				}
			}
			int y = curry - 13;
			int x_start = currx + 4;
			int x_end = currx - 9;
			for (y; y - curry >= -18; y--)
			{

				for (int x = x_start; x >= x_end; x--)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				x_start--;
				x_end--;
			}

			int y_new = curry - 19;
			x_start = currx - 3;
			x_end = currx - 15;

			for (y_new; y_new - curry >= -24; y_new--)
			{

				for (int x = x_start; x >= x_end; x--)
				{
					if (!map->isTransFree(x, y))
					{
						return 0; //collision
					}
				}
				x_start--;
				x_end++;
			}
			return 1;
		}

		//for heading 270 and backwards
		else if (dir == 3 && heading == 270)
		{
			for (int i = curry + 8; i <= curry + 9; i++)
			{
				for (int j = currx - 4; j <= currx + 4; j++)
				{
					if (!map->isTransFree(j, i))
					{
						return 0; //collision if map is not free
					}
				}
			}
			return 1;
		}

		// for heading 270 few steps forward
		else if (dir == 4 && heading == 270)
		{
			for (int i = curry - 8; i >= curry - fwd_step - 9; i--)
			{
				for (int j = currx - 4; j <= currx + 4; j++)
				{
					if (!map->isTransFree(j, i))
					{
						return 0; //collision if map is not free
					}
				}
			}
			return 1;
		}

		//for heading 315 and forward
		else if (dir == 0 && heading == 315)
		{
			int j = curry - 10;
			for (int i = currx + 4; i <= currx + 10; i++)
			{
				if (!map->isTransFree(i, j))
				{
					return 0; //collision
				}
				j++;
			}
			return 1;
		}

		//for heading 315 and ccw turn
		else if (dir == 1 && heading == 315)
		{
			for (int i = currx + 3; i <= currx + 23; i++)
			{
				for (int j = curry - 2; j >= curry - 10; j--)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		//for heading 315 and cw turn
		else if (dir == 2 && heading == 315)
		{
			for (int i = currx + 2; i <= currx + 10; i++)
			{
				for (int j = curry - 4; j >= curry - 23; j--)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
				}
			}
			return 1;
		}

		//for heading 315 and backwards
		else if (dir == 3 && heading == 315)
		{
			int j = curry + 10;
			for (int i = currx - 4; i >= currx - 10; i--)
			{
				if (!map->isTransFree(i, j))
				{
					return 0; //collision
				}
				j--;
			}
			return 1;
		}

		//for heading 315 and few steps forward
		else if (dir == 4 and heading == 315)
		{
			for (currx; currx <= currx + fwd_step; currx++)
			{
				int j = curry - 10;
				for (int i = currx + 4; i <= currx + 10; i++)
				{
					if (!map->isTransFree(i, j))
					{
						return 0; //collision
					}
					j++;
				}
				curry--;
			}
			return 1;
		}
	}


	
};
