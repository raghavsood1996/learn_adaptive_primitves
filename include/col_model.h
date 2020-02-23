#include <torch/script.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>



bool reed_net_col(std::vector<double> lidar_scan, torch::jit::script::Module* module) {

    
    torch::Tensor feat = torch::ones({1,362});
   
    int count = 0;
    while(count <362){

        feat[0][count] = lidar_scan[count];
        //cout<<lidar_scan[count]<<" "<<" feat "<<feat[0][count]<<" "<<count<<std::endl;
        count++;
    }
    
   
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(feat);
    at::Tensor output = module->forward(inputs).toTensor();
    //cout<<"network output"<<output.item<float>()<<endl;
    if(output.item<double>() >= 0.5){
        
        return 1;
    }

    return 0;
}