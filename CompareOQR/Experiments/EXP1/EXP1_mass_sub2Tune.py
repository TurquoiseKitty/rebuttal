from trainer import grid_searcher
from trainer2 import grid_searcher_v2
import os
import yaml

if __name__ == "__main__":


    for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:
        with open(os.getcwd()+"/Experiments/EXP1/config_bin/vanillaPred_on_"+dataname+"_config.yml", 'r') as file:
            base_configs = yaml.safe_load(file)

        print("model: "+ "vanillaMSQR" +" on data: "+dataname)
        grid_searcher_v2(
            dataset_name= dataname,
            num_repeat=5,
            model_name = "vanillaMSQR",
            to_search = {
                "wid" : [5, 10],
                "LR": [5E-3]
                },
            base_misc_info = base_configs["misc_info"],
            base_train_config= base_configs["training_config"],
            aux_config = {},
        )




    for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:

        modelname = "GPmodel"
        print("model: "+ modelname +" on data: "+dataname)
        grid_searcher(
            dataset_name = dataname,
            num_repeat = 5,
            model_name = modelname,
            to_search = {
                "LR": [1E-2, 5E-3],
                "bat_size": [64]
            }
        )


