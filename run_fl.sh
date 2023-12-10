#!/bin/bash -ex
python main_her2_FL.py --exp_code central_baseline --no_fl
python main_her2_FL.py --fl_avg FedAvg --exp_code PointTransformer
python main_her2_FL.py --fl_avg FedAvg --exp_code PointTransformer+ --fast_sim
python main_her2_FL.py --fl_avg FedAvg --exp_code PointTransformerDDA --aux 1.0
python main_her2_FL.py --fl_avg FedAvg --exp_code PointTransformerDDA+ --aux 1.0 --fast_sim