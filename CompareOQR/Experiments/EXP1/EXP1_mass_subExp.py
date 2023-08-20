from trainer import grid_searcher
from trainer2 import grid_searcher_v2
import os
import yaml

if __name__ == "__main__":

    for modelname in [ "HNN", "MC_drop", "DeepEnsemble", "HNN_BeyondPinball", "vanillaPred"]:

        for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:


            print("model: "+ modelname +" on data: "+dataname)
            grid_searcher(
                dataset_name = dataname,
                num_repeat = 5,
                model_name = modelname,
                to_search = {
                    "LR": [1E-2, 5E-3],
                    "bat_size": [10, 64]
                }
            )


    for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:
         
        modelname = "HNN_MMD"
        print("model: "+ modelname +" on data: "+dataname)

        with open(os.getcwd()+"/Experiments/EXP1/config_bin/HNN_on_"+dataname+"_config.yml", 'r') as file:
            base_configs = yaml.safe_load(file)

        
        grid_searcher_v2(
            dataset_name= dataname,
            num_repeat=5,
            model_name = "HNN_MMD",
            to_search = {
                "LR" : [5E-3, 1E-3],
                "bat_size": [64, 128]
            },
            base_misc_info = base_configs["misc_info"],
            base_train_config= base_configs["training_config"],
            aux_config = {},
        )


    for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:

        with open(os.getcwd()+"/Experiments/EXP1/config_bin/vanillaPred_on_"+dataname+"_config.yml", 'r') as file:
            base_configs = yaml.safe_load(file)

        grid_searcher_v2(
            dataset_name= dataname,
            num_repeat=5,
            model_name = "vanillaKernel",
            to_search = {"wid" : [5, 1, 5E-1, 1E-1, 5E-2, 1E-2] },
            base_misc_info = base_configs["misc_info"],
            base_train_config= base_configs["training_config"],
            aux_config = {},
        )
