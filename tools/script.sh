python generate_grid_scenario.py 1 --rowDistance 600 --columnDistance 600 --intersectionWidth 20 --numLeftLanes 0 --numRightLanes 0 --tlPlan
test/Simulator_test --roadnetFile roadnet_1_1.json --flowFile flow_1_1.json --totalStep 1000 --roadnetLogFile roadnet_1_1.json --replayLogFile replay_1_1.txt --saveReplay --verbose