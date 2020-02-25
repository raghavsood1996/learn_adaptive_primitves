#pragma once

#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <time.h>
#include <random>
#include <chrono>
#include <fssimplewindow.h>
#include <Bitmap.h>
#include <motion_prim.h>
#include <planner.h>
#include <dubins.h>
#include <KDtree.hpp>
#include <chrono>
//#include "reeds_shepp.h"
struct query{
	node start;
	node goal;
	int map_id;

	query(node start,node goal, int map_id){
		this->start = start;
		this->goal = goal;
		this->map_id = map_id;
	}
};

double deg2rad(int degree)
{
	return degree * 3.142 / 180;
}

config sample_random_config(int fix_heading_idx = -1)
{
	random_device rnd;
	mt19937 mt(rnd());
	mt19937 mt2(rnd());
	int heading[] = {0, 45, 90, 135, 180, 225, 270, 315};

	uniform_int_distribution<int> heading_idx_dist(0, 7);
	uniform_int_distribution<int> x_idx_dist(0, 400);
	uniform_int_distribution<int> y_idx_dist(0, 400);
	auto x_idx = bind(x_idx_dist, mt);
	auto y_idx = bind(y_idx_dist, mt2);
	auto heading_idx = bind(heading_idx_dist, mt);
	if(fix_heading_idx == -1){
		return config(x_idx(), y_idx(), heading[heading_idx()]);
	}

	else{
		return config(x_idx(), y_idx(), heading[fix_heading_idx]);
	}


}

void write_config_to_file(ofstream &out_file, config samp_conf){

	out_file << to_string(samp_conf.x)<<','<<to_string(samp_conf.y)<<"\n";

	return;
	
}

config sample_config_hval(state2d **states, int max_hval, node &goal, int fixed_heading_idx = -1)
{
	random_device rnd;
	mt19937 mt(rnd());
	mt19937 mt2(rnd());
	int heading[] = {0, 45, 90, 135, 180, 225, 270, 315};
	uniform_int_distribution<int> heading_idx_dist(0, 7);
	int min_x, max_x, min_y, max_y;

	if(goal.x + max_hval<390){
		max_x = goal.x + max_hval;
	}
	else{
		max_x = 390;
	}

	if(goal.y + max_hval<390){
		max_y = goal.y + max_hval;
	}
	else{
		max_y = 390;
	}

	if(goal.x - max_hval>11){
		min_x = goal.x - max_hval;
	}
	else{
		min_x = 11;
	}

	if(goal.y - max_hval > 11){
		min_y = goal.y - max_hval;
	}
	else{
		min_y = 11;
	}

	uniform_int_distribution<int> x_idx_dist(min_x, max_x);
	uniform_int_distribution<int> y_idx_dist(min_y, max_y);
	auto x_idx = bind(x_idx_dist, mt);
	auto y_idx = bind(y_idx_dist, mt2);
	auto heading_idx = bind(heading_idx_dist, mt);
	config sample_config;

	do
	{	if(fixed_heading_idx == -1){
		sample_config = config(x_idx(), y_idx(), heading[heading_idx()]);
		}

		else{
		sample_config = config(x_idx(), y_idx(), heading[fixed_heading_idx]);
		}
	} while (states[x_idx()][y_idx()].gval > max_hval);

	return sample_config;
}


config sample_random_local(config curr, int radius){
	random_device rnd;
	mt19937 mt(rnd());
	mt19937 mt2(rnd());
	int heading[] = {0, 45, 90, 135, 180, 225, 270, 315};
	uniform_int_distribution<int> heading_idx_dist(0, 7);
	int min_x, max_x, min_y, max_y;

	if(curr.x + radius < 390){
		max_x = curr.x + radius;
	}
	else{
		max_x = 390;
	}

	if(curr.y + radius < 390){
		max_y = curr.y + radius;
	}
	else{
		max_y = 390;
	}

	if(curr.x - radius>11){
		min_x = curr.x - radius;
	}
	else{
		min_x = 11;
	}

	if(curr.y - radius > 11){
		min_y = curr.y - radius;
	}
	else{
		min_y = 11;
	}
	
	uniform_int_distribution<int> x_idx_dist(min_x, max_x);
	uniform_int_distribution<int> y_idx_dist(min_y, max_y);
	auto x_idx = bind(x_idx_dist, mt);
	auto y_idx = bind(y_idx_dist, mt2);
	auto heading_idx = bind(heading_idx_dist, mt);
	config sample_config;

	
	
	sample_config = config(x_idx(), y_idx(), heading[heading_idx()]);
	

	return sample_config;
}
node sample_random_node()
{
	random_device rnd;
	mt19937 mt(rnd());
	mt19937 mt2(rnd());
	int heading[] = {0, 45, 90, 135, 180, 225, 270, 315};
	uniform_int_distribution<int> heading_idx_dist(0, 7);
	uniform_int_distribution<int> x_idx_dist(11, 385);
	uniform_int_distribution<int> y_idx_dist(11, 385);
	auto x_idx = bind(x_idx_dist, mt);
	auto y_idx = bind(y_idx_dist, mt2);
	auto heading_idx = bind(heading_idx_dist, mt);
	node sample_node(x_idx(), y_idx(), heading[heading_idx()]);

	return sample_node;
}



//map precomputation without using the neural net
void no_net_precomp(state2d **state, CharBitmap *map, unordered_map<config, bool, confighasher, configComparator> &reed_map, node goal_node,array<array<float, 3>, SAMPLES> &tree_data, int num_states){

	std::vector<config> pre_configs;
	double t_time = 0;
	int states = 0;
	clock_t begin = clock();
	int fix_start_heading = 1;
	while (states < num_states)
	{		
		config temp;
		if (states < 9000) //sampling in environment
		{
			temp = sample_random_config(fix_start_heading); //sampling a random config
			if (!map->isFree(temp.x, temp.y))
			{
				continue;
			}
			pre_configs.push_back(temp);
		}
		else
		{
			temp = sample_config_hval(state, 100, goal_node,fix_start_heading); //sampling around goal
			pre_configs.push_back(temp);
		}

		tree_data[states] = {temp.x, temp.y, deg2rad(temp.theta)};

		states++;
	}

	int config_count = 0;
	
	ofstream valid_ofile;
	ofstream invalid_ofile;
	valid_ofile.open("../stats/valid_states.txt");
	invalid_ofile.open("../stats/invalid_states.txt");

	for (int i = 0; i < states; i++)
	{

		double curr[] = {pre_configs[config_count].x, pre_configs[config_count].y, pre_configs[config_count].theta* 3.142 / 180.0};
		double goal[] = {goal_node.x, goal_node.y, goal_node.theta * 3.142 / 180.0};
		double turning_radius = 22.0;
		ReedsSheppStateSpace space(turning_radius);
		vector<config> path_samples;
		ReedsSheppStateSpace::ReedsSheppPath path;
		path = space.reedsShepp(curr, goal);
		space.sample(&path_samples, curr, goal, 2, reeds_cb, NULL);

		if (dubins_collision_check(path_samples, map))
		{
			reed_map[pre_configs[config_count]] = 1;
			write_config_to_file(valid_ofile,pre_configs[config_count]);
		}

		else 
		{
			reed_map[pre_configs[config_count]] = 0;
			write_config_to_file(invalid_ofile,pre_configs[config_count]);
		}
		config_count++;
	}
	valid_ofile.close();
	invalid_ofile.close();
	t_time = ((float(clock() - begin)) / CLOCKS_PER_SEC);
	cout << "precompute time " << t_time << endl;
}

void biased_sampling_precomp(state2d **state, CharBitmap *map, torch::jit::script::Module &mod, unordered_map<config, bool, confighasher, configComparator> &reed_map, node goal,
					array<array<float, 3>, SAMPLES> &tree_data, int num_states){
	std::vector<torch::Tensor> inputs; //all network inputs stacked
	std::vector<torch::jit::IValue> final_inputs;
	std::vector<config> pre_configs;
	int num_rays = 360;
	int step = 8;
	double t_time = 0;
	int states = 0;
	using clock = std::chrono::system_clock;
	using ms = std::chrono::duration<double, std::milli>;
	int fix_init_samp = 1000;
	while(states < fix_init_samp){
		config temp;
		
		temp = sample_random_config(); //sampling a random config
		if (!map->isFree(temp.x, temp.y)) continue;
		pre_configs.push_back(temp);
		tree_data[states] = {temp.x, temp.y, deg2rad(temp.theta)};
		
		//torch::Tensor obst_distances = map->dist_nearest_obst(temp, num_rays, step); //generating the feature vector from ray tracing
		torch::Tensor obst_distances2 = map->ray_tracing_2(temp, 360,8,0,8);
		obst_distances2[0][num_rays / step] = goal.theta - temp.theta;
		obst_distances2[0][(num_rays / step) + 1] = distance(temp.x, temp.y, goal.x, goal.y);
		inputs.push_back(obst_distances2);
		
		
		states++;
	}					

	at::Tensor comb_input = torch::cat(inputs); //concatenating all the input tensors in a batch
	final_inputs.push_back(comb_input); // the final input to the network
	const auto before = clock::now();
	at::Tensor output = mod.forward(final_inputs).toTensor(); //inference
	const ms duration = clock::now() - before;
	cout << "forward pass precompute time for "<<SAMPLES<<" samples " << duration.count()<<" ms" << endl;

	
	int config_count = 0;
	
	
	for (int i = 0; i < states; i++)
	{

		if (output[i].item<float_t>() > 0.5)
		{
			reed_map[pre_configs[config_count]] = 1;
		}
		else
		{
			reed_map[pre_configs[config_count]] = 0;
		}
		config_count++;
	}



	while(states < num_states){
		for(auto i = reed_map.begin(); i != reed_map.end(); i++){
			auto sam_conf = i->first;
			if(!i->second){
				continue;
			}

			for(int j=0;j< 4; j++){
				config temp = sample_random_local(sam_conf,60);

				if (!map->isFree(temp.x, temp.y)) continue;
				pre_configs.push_back(temp);
				// cout<<"done second"<<"\n";
				// cout<<states<<"\n";
				tree_data[states] = {temp.x, temp.y, deg2rad(temp.theta)};
				
				torch::Tensor obst_distances2 = map->ray_tracing_2(temp, 360,8,0,8);
				obst_distances2[0][num_rays / step] = goal.theta - temp.theta;
				obst_distances2[0][(num_rays / step) + 1] = distance(temp.x, temp.y, goal.x, goal.y);
				inputs.push_back(obst_distances2);
		
		
				states++;
				if(states > SAMPLES){
				break;
			}
			}

			if(states > SAMPLES){
				break;
			}


		}
	}


	comb_input = torch::cat(inputs); //concatenating all the input tensors in a batch
	final_inputs.clear();
	final_inputs.push_back(comb_input); // the final input to the network
	
	output = mod.forward(final_inputs).toTensor(); //inference
	// const ms duration = clock::now() - before;
	// cout << "forward pass precompute time for "<<SAMPLES<<" samples " << duration.count()<<" ms" << endl;

	
	config_count = 0;
	
	
	for (int i = 0; i < states; i++)
	{

		if (output[i].item<float_t>() > 0.5)
		{
			reed_map[pre_configs[config_count]] = 1;
		}
		else
		{
			reed_map[pre_configs[config_count]] = 0;
		}
		config_count++;
	}

					
}
//precompute the map using neural network
void precompute_map(state2d **state, CharBitmap *map, torch::jit::script::Module &mod, unordered_map<config, bool, confighasher, configComparator> &reed_map, node goal,
					array<array<float, 3>, SAMPLES> &tree_data, int num_states)
{	
	std::vector<torch::Tensor> inputs; //all network inputs stacked
	std::vector<torch::jit::IValue> final_inputs;
	std::vector<config> pre_configs;
	int num_rays = 360;
	int step = 8;
	double t_time = 0;
	int states = 0;
	using clock = std::chrono::system_clock;
	using ms = std::chrono::duration<double, std::milli>;
	
	while (states < num_states)
	{
		
		config temp;
		if (states < 4000) //sampling in a config environment
		{
			temp = sample_random_config(); //sampling a random config
			if (!map->isFree(temp.x, temp.y)) continue;
		}
		else
		{
			temp = sample_config_hval(state, 100, goal); //sampling around goal
			
		}
		pre_configs.push_back(temp);
		tree_data[states] = {temp.x, temp.y, deg2rad(temp.theta)};
		
		//torch::Tensor obst_distances = map->dist_nearest_obst(temp, num_rays, step); //generating the feature vector from ray tracing
		torch::Tensor obst_distances2 = map->ray_tracing_2(temp, 360,8,0,8);
		obst_distances2[0][num_rays / step] = goal.theta - temp.theta;
		obst_distances2[0][(num_rays / step) + 1] = distance(temp.x, temp.y, goal.x, goal.y);
		inputs.push_back(obst_distances2);
		
		
		states++;
	}
	
	
	at::Tensor comb_input = torch::cat(inputs); //concatenating all the input tensors in a batch
	final_inputs.push_back(comb_input); // the final input to the network
	const auto before = clock::now();
	at::Tensor output = mod.forward(final_inputs).toTensor(); //inference
	const ms duration = clock::now() - before;
	cout << "forward pass precompute time for "<<SAMPLES<<" samples " << duration.count()<<" ms" << endl;

	
	int config_count = 0;
	
	// storing everything in the table for lookup
	for (int i = 0; i < states; i++)
	{

		if (output[i].item<float_t>() > 0.5)
		{
			reed_map[pre_configs[config_count]] = 1;
		}
		else
		{
			reed_map[pre_configs[config_count]] = 0;
		}
		config_count++;
	}

	
	
}

//Draws the dubins path on the screen
void DrawReeds(std::vector<config> path_points)
{
	glColor3f(1, 1, 1);
	glBegin(GL_LINE_STRIP);
	for (int i = 0; i < path_points.size(); i++)
	{
		glVertex2f(path_points[i].x * 4, (400 - path_points[i].y) * 4);
	}
	glEnd();
}

//scales the dubins path to match drawing scale of the screen
std::vector<config> ScaleReeds(std::vector<config> path_points, int scale)
{
	std::vector<config> scaled_path;
	for (config c : path_points)
	{
		config temp(int(c.x * scale), int(400 - c.y) * scale, c.theta);
		scaled_path.push_back(temp);
	}
	return scaled_path;
}

//callback function for dubins
// int reeds_cb(std::vector<config> *path_samples, double q[3], void *user_data)
// {
// 	config temp(q[0], q[1], q[2]);
// 	path_samples->push_back(temp);
// 	return 0;
// }

double rad2deg(int rad)
{

	return (rad * 180 / 3.14);
}

void draw_whole_path(std::vector<config> path)
{

	glColor3ub(100, 150, 0);
	glBegin(GL_LINE_STRIP);
	for (config c : path)
	{
		glVertex2f(c.x, c.y);
		cout << " path x " << c.x;
		cout << " path y " << c.y;
		cout << " path angle " << rad2deg(c.theta) << endl;
	}
	glEnd();
}

// Function that draws the generated path on the screen
std::vector<config> *DrawPath(std::vector<node> plan, CharBitmap *map)
{
	//declare a std::vector thar will store the path
	std::vector<config> *traj = new std::vector<config>;

	for (int i = 0; i < plan.size() - 1; i++)
	{
		int plan_idx = plan[i + 1].theta - plan[i].theta;
		int prim_idx = 0;
		if (plan_idx == 0 || plan_idx == 360 || plan_idx == -360)
		{
			if (plan[i + 1].pre_cost < 1.43)
			{
				prim_idx = 0; //moving a step forward
			}
			else if (plan[i + 1].pre_cost >= 8 && plan[i + 1].pre_cost <= 8 * 1.43)
			{
				prim_idx = 4; //moving a few steps forward
			}
			else
			{
				prim_idx = 3; //moving backwards
			}
		}
		else if (plan_idx == 45 || plan_idx == -315)
		{
			prim_idx = 1; //ccw turn
		}
		else if (plan_idx == -45 || plan_idx == 315)
		{
			prim_idx = 2; //cw turn
		}

		//drawing dubins path if it is part of the plan
		else
		{

			double curr[] = {double(plan[i].x), double(plan[i].y), plan[i].theta * 3.14 / 180};
			double goal[] = {double(plan[i + 1].x),double( plan[i + 1].y), plan[i + 1].theta * 3.14 / 180};

			std::vector<config> scaled_path;

			double turning_radius = 22.0;
			ReedsSheppStateSpace space(turning_radius);
			std::vector<config> path_samples;
			ReedsSheppStateSpace::ReedsSheppPath path;
			path = space.reedsShepp(curr, goal);
			space.sample(&path_samples, curr, goal, 2, reeds_cb, NULL);
			scaled_path = ScaleReeds(path_samples, 4);
			DrawReeds(path_samples);
		}

		//map->DrawCar(plan[i].x, plan[i].y, plan[i].theta,"empty");
		map->DrawTrajectory(traj, plan[i].x, plan[i].y, plan[i].theta, prim_idx);
	}
	return traj;
}

//util function for claculating area
float area(int x1, int y1, int x2, int y2,
		   int x3, int y3)
{
	return abs((x1 * (y2 - y3) + x2 * (y3 - y1) +
				x3 * (y1 - y2)) /
			   2.0);
}

bool check_point(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, int x, int y)
{
	//cout << "point being checked is" << x << " " << y << endl;
	/* Calculate area of rectangle ABCD */
	float A = area(x1, y1, x2, y2, x3, y3) +
			  area(x1, y1, x4, y4, x3, y3);

	/* Calculate area of triangle PAB */
	float A1 = area(x, y, x1, y1, x2, y2);

	/* Calculate area of triangle PBC */
	float A2 = area(x, y, x2, y2, x3, y3);

	/* Calculate area of triangle PCD */
	float A3 = area(x, y, x3, y3, x4, y4);

	/* Calculate area of triangle PAD */
	float A4 = area(x, y, x1, y1, x4, y4);

	/* Check if sum of A1, A2, A3 and A4  
       is same as A */
	//cout << "A" << A << endl;
	//cout << "triangles_combined" << A1 + A2 + A3 + A4 << endl;
	return (A == A1 + A2 + A3 + A4);
}

//checks if the requested config is valid
bool isConfigValid(CharBitmap *env, node state)
{

	double angle = -state.theta * 3.14 / 180;

	//coordinates of the quadrilateral
	int x1 = state.x + (8) * cos(angle) - (4) * sin(angle);
	int y1 = state.y + (8) * sin(angle) + (4) * cos(angle);

	int x2 = state.x + (-8) * cos(angle) - (4) * sin(angle);
	int y2 = state.y + (-8) * sin(angle) + (4) * cos(angle);

	int x3 = state.x + (-8) * cos(angle) - (-4) * sin(angle);
	int y3 = state.y + (-8) * sin(angle) + (-4) * cos(angle);

	int x4 = state.x + (8) * cos(angle) - (-4) * sin(angle);
	int y4 = state.y + (8) * sin(angle) + (-4) * cos(angle);

	//cout << x1 << " " << y1 << " " << x2 << " " << y2 << " " << x3 << " " << y3 << " " << x4 << " " << y4 << endl;
	//cout << start_x << " " << start_y << endl;

	for (int start_x = state.x - 8; start_x <= state.x + 8; start_x++)
	{

		for (int start_y = state.y - 8; start_y <= state.y + 8; start_y++)
		{

			if (!env->isFree(start_x, start_y) /*&& check_point(x1,y1,x2,y2,x3,y3,x4,y4,start_x,start_y)*/)
			{
				//cout << "not free point found..now checking if its inside the car" << endl;
				if (check_point(x1, y1, x2, y2, x3, y3, x4, y4, start_x, start_y))
				{
					cout << "invalid configuration demanded" << endl;
					return 0;
				}
			}
		}
	}
	cout << "valid_config" << endl;
	return 1;
}


//saves the configuration to a text file
void save_nodes()
{

	ofstream myfile("example.txt");
	if (myfile.is_open())
	{
		myfile << "This is a line.\n";
		myfile << "This is another line.\n";
		myfile.close();
	}
	else
		cout << "Unable to open file";
	
	return;
}

node string_to_node(string query){
	string x ="";
	string y ="";
	string theta ="";

	for(int i = 0; i<query.length(); i++){
		if(query[i]==','){
			query = query.substr(i+1,query.length()-1);
			break;
		}

		x+=query[i];
	}

	
	for(int i = 0; i<query.length(); i++){
		if(query[i]==','){
			query = query.substr(i+1,query.length()-1);
			break;
		}

		y+=query[i];
	}

	

	for(int i = 0; i<query.length(); i++){
		if(query[i]==','){
			query = query.substr(i+1,query.length()-1);
			break;
		}

		theta+=query[i];
	}

	node temp(stoi(x),stoi(y),stoi(theta));
	return temp;

}

//read search queries from a text file
vector<query> read_queries(string filename){

	vector<query> inputs;
	vector<string> vecs;
	ifstream in_file("queries.txt");
	string str;
	while(getline(in_file,str)){
		vecs.push_back(str);
	}

	for(int i=0; i<vecs.size()-2; i+=3){
		int map_num = stoi(vecs[i]);
		node start = string_to_node(vecs[i+1]);
		node goal = string_to_node(vecs[i+2]);
		query temp(start,goal,map_num);
		inputs.push_back(temp);
	}
	

	return inputs;
}