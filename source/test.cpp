#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <time.h>
#include <random>
#include <chrono>
#include <torch/script.h>
#include <fstream>
#include <fssimplewindow.h>
#include <Bitmap.h>
#include <motion_prim.h>
#include <planner.h>
#include <utils.h>
#include <map>
#include <array>

using namespace std;

int test()
{
	CharBitmap bitu; // the cost map
	CharBitmap *map_ptr;
	map_ptr = &bitu;
	lattice_graph *thegraph = new lattice_graph; //The lattice graph
	thegraph->set_motion_prims();

	// node start(295,268,270);  //Start pose for the vehicle
	// node goal(111,25,90); //Goal pose for the vehicle
	//save_nodes();
	node start;
	node goal;

	int w, h, key;
	bool terminate = false;

	w = 400;
	h = 400;

	bitu.create(w, h);

	int scale = 4;

	int lb, mb, rb, sx, sy;
	int x, y;
	string filename;
	vector<plan_stats> all_stats_net;
	vector<plan_stats> all_stats_no_net;

	//Loading the Network
	torch::jit::script::Module mod = torch::jit::load("/home/raghav/Raghav/Research/xyt_domain/ML_model2/trained_networks/cpp_model360_8.pt"); 
	random_device rnd;
	mt19937 mt(rnd());
	vector<node> plan_net;
	vector<node> plan_no_net;
	uniform_int_distribution<int> map_num(1, 100);
	auto map_idx = bind(map_num, mt);

	//ofstream out_file("queries_10_4.txt");
	vector<query> rand_queries = read_queries("queries.txt");

	for (query q : rand_queries /*int i=0; i<5; i++*/)
	{
		unordered_map<config, bool, confighasher, configComparator> reed_map; //map that stores the output from net
		const int precom_samp = 5000;										  //number of samples to precompute for
		array<array<float, 3>, precom_samp> tree_data;
		int rnd_map_id = map_idx();
		plan_stats *curr_stat_net = new plan_stats;
		plan_stats *curr_stat_no_net = new plan_stats;

		//filename = "../maps/map"+to_string(rnd_map_id)+".txt";
		//filename = "env"+to_string(rnd_map_id)+".txt";

		filename = "../maps/map" + to_string(q.map_id) + ".txt";
		cout << "Environment " << filename << endl;
		bitu.load(filename);

		map_transform(map_ptr); //Inflate obstacles
		
		start = q.start;
		goal = q.goal;
		goal.hval = 0;

		cout << "start " << start << endl;
		cout << "goal" << goal << endl;

		state2d **states = new state2d *[400]; //Dynamic array that stores G values from backward Djiktras search
		for (int i = 0; i < 400; ++i)
		{
			states[i] = new state2d[400];
		}
		
		
		heuristic_planner(states, map_ptr, goal); //The planner that calculates heuristics using a backward Djiktras search

		precompute_map(states, map_ptr, mod, reed_map, goal, tree_data, precom_samp);
		//no_net_precomp(states,map_ptr,reed_map,goal,tree_data, precom_samp);
		clock_t begin = clock();
		KDtree<float, precom_samp, 3> kdtree(&tree_data);
		cout << "Kd tree precompute time " << ((float(clock() - begin)) / CLOCKS_PER_SEC) << endl;

		plan_net = planner(states, map_ptr, thegraph, start, goal, curr_stat_net, reed_map, kdtree, tree_data, false);
		all_stats_net.push_back(*curr_stat_net);

		// if(curr_stat_net->plan_cost != 0){

		// 	out_file << to_string(rnd_map_id)+"\n";
		// 	out_file << start.toString()+"\n";
		// 	out_file << goal.toString()+"\n";

		// }

		//plan_no_net = planner(states, map_ptr, thegraph, start, goal, curr_stat_no_net,reed_map,kdtree,tree_data,false);
		//all_stats_no_net.push_back(*curr_stat_no_net);
		delete curr_stat_net;
		delete curr_stat_no_net;
	}

	//out_file.close();

	double average_plan_time_net = 0;
	double average_cost_net = 0;
	double average_exp_net = 0;
	double max_plan_time_net = 0;
	double std_dev_net = 0;

	for (plan_stats stat : all_stats_net)
	{
		average_plan_time_net += stat.planning_time;
		if (stat.planning_time > max_plan_time_net)
		{
			max_plan_time_net = stat.planning_time;
		}
		average_cost_net += stat.plan_cost;
		average_exp_net += stat.exp_per_sec;
	}

	average_plan_time_net /= all_stats_net.size();
	average_cost_net /= all_stats_net.size();
	average_exp_net /= all_stats_net.size();

	for (plan_stats stat : all_stats_net)
	{
		double sq_diff = pow(stat.planning_time - average_plan_time_net, 2);
		std_dev_net += sq_diff;
	}
	std_dev_net /= all_stats_net.size();

	cout << "stats for planner using the Net " << endl;
	cout << "Average Time " << average_plan_time_net << endl;
	cout << "Maximum Planning Time " << max_plan_time_net << endl;
	cout << " Standard deviation " << std_dev_net << endl;
	cout << "Average Cost " << average_cost_net << endl;
	cout << "Average Expansions per Second :" << average_exp_net << endl;

	double average_plan_time_no_net = 0;
	double average_cost_no_net = 0;
	double average_exp_no_net = 0;
	double max_plan_time_no_net = 0;
	double std_dev_no_net = 0;

	for (plan_stats stat : all_stats_no_net)
	{
		average_plan_time_no_net += stat.planning_time;
		if (stat.planning_time > max_plan_time_no_net)
		{
			max_plan_time_no_net = stat.planning_time;
		}
		average_cost_no_net += stat.plan_cost;
		average_exp_no_net += stat.exp_per_sec;
	}

	average_plan_time_no_net /= all_stats_no_net.size();
	average_cost_no_net /= all_stats_no_net.size();
	average_exp_no_net /= all_stats_no_net.size();

	for (plan_stats stat : all_stats_no_net)
	{
		double sq_diff = pow(stat.planning_time - average_plan_time_no_net, 2);
		std_dev_no_net += sq_diff;
	}
	std_dev_no_net /= all_stats_no_net.size();

	cout << endl;
	cout << "Stats for planner without using Network" << endl;
	cout << "Average Time " << average_plan_time_no_net << endl;
	cout << "Maximum Planning Time " << max_plan_time_no_net << endl;
	cout << " Standard deviation " << std_dev_no_net << endl;
	cout << "Average Cost " << average_cost_no_net << endl;
	cout << "Average Expansions per Second :" << average_exp_no_net << endl;

	FsOpenWindow(0, 0, w * scale, h * scale, 1);

	int draw_itr = 0;

	while (!terminate)
	{
		FsPollDevice();
		key = FsInkey();
		FsGetMouseEvent(lb, mb, rb, sx, sy);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		// Add real time obstacles to the map
		if (FSKEY_0 <= key && key <= FSKEY_7)
		{
			x = sx / scale;
			y = sy / scale;
			bitu.setPixel(x, y, '0' + key - FSKEY_0);
		}
		else
		{
			switch (key)
			{
			case FSKEY_ESC:
				terminate = true;

				// Save a map
			case FSKEY_S:
				cout << "Save File Name? ";
				cin >> filename;
				bitu.save(filename);
				break;

				//Load a saved map
			case FSKEY_L:
				cout << "Load file Name? ";
				cin >> filename;
				bitu.load(filename);
				break;
			}
		}

		bitu.draw();

		DrawPath(plan_net, map_ptr); //changed the path function
		//DrawPath(plan_no_net,map_ptr);
		//bitu.DrawCar(sample[draw_itr].x, sample[draw_itr].y,rad2deg(sample[draw_itr].theta), "filled");
		bitu.DrawCar(goal.x, goal.y, goal.theta, "filled");
		bitu.DrawCar(start.x, start.y, start.theta, "");

		draw_itr++;
		if (draw_itr == plan_no_net.size() - 1)
		{
			draw_itr = 0;
		}

		FsSwapBuffers();
		FsSleep(700);
	}
	return 0;
}