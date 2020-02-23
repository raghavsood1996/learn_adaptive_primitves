#pragma once
#include <utils.h>

void valid_prim_data(CharBitmap *map, unordered_map<config, bool, confighasher, configComparator> &reed_map, node goal_node){

	std::vector<config> pre_configs;
	double t_time = 0;
	int states = 0;
	clock_t begin = clock();
	
	
    for(int i=0; i<399; i++){
        for(int j=0; j<399; j++){
            config temp(j,i,270);
            pre_configs.push_back(temp);
            states++;
        }
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


void predict_prim_data(CharBitmap *map, torch::jit::script::Module &mod, unordered_map<config, bool, confighasher, configComparator> &reed_map, node goal,
					array<array<float, 3>, SAMPLES> &tree_data)
{	
	std::vector<torch::Tensor> inputs(16000); //all network inputs stacked
	std::vector<torch::jit::IValue> final_inputs;
	std::vector<config> pre_configs;
	int num_rays = 360;
	int step = 8;
	double t_time = 0;
	int states = 0;
	using clock = std::chrono::system_clock;
	using ms = std::chrono::duration<double, std::milli>;
	const auto before = clock::now();



	for(int i=0; i<400; i++){
        for(int j=0; j<400; j++){
            config temp(j,i,0);
			pre_configs.push_back(temp);
			tree_data[states] = {temp.x, temp.y, deg2rad(temp.theta)};
			
			//torch::Tensor obst_distances = map->dist_nearest_obst(temp, num_rays, step); //generating the feature vector from ray tracing
			torch::Tensor obst_distances2 = map->ray_tracing_2(temp, 360,8,0,8);
			obst_distances2[0][num_rays / step] = goal.theta - temp.theta;
			obst_distances2[0][(num_rays / step) + 1] = distance(temp.x, temp.y, goal.x, goal.y);
			inputs[states]= obst_distances2;
			states++;
			cout<<states<<"\n";
			
        }
		
    }
	
	
	at::Tensor comb_input = torch::cat(inputs); //concatenating all the input tensors in a batch
	final_inputs.push_back(comb_input); // the final input to the network

	at::Tensor output = mod.forward(final_inputs).toTensor(); //inference
	
	int config_count = 0;
	ofstream valid_ofile;
	ofstream invalid_ofile;
	valid_ofile.open("../stats/predicted_valid_states.txt");
	invalid_ofile.open("../stats/predicted_invalid_states.txt");
	
	// storing everything in the table for lookup
	for (int i = 0; i < states; i++)
	{

		if (output[i].item<float_t>() > 0.5)
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

	const ms duration = clock::now() - before;
	cout << "forward pass precompute time " << duration.count()<<" ms" << endl;

	
}
