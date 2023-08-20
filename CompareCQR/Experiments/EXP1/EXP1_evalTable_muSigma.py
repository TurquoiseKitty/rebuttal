
import os
import yaml
from trainer import trainer, model_callByName, loss_callByName
from data_utils import get_uci_data, common_processor_UCI, seed_all
from TestPerform import testPerform_muSigma
import torch
import pandas as pd


if __name__ == "__main__":

    SEED = 1234
    num_repeat = 5
    big_df = {}

    for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:

        err_mu_dic = {}
        err_std_dic = {}

        dataset_path = os.getcwd()+"/Dataset/UCI_datasets"

        x, y = get_uci_data(data_name= dataname, dir_name= dataset_path)

        for modelname in [ "HNN", "MC_drop", "DeepEnsemble", "HNN_BeyondPinball"]:

            # train base model
            print("model: "+ modelname +" on data: "+dataname)

            with open(os.getcwd()+"/Experiments/EXP1/config_bin/"+modelname+"_on_"+dataname+"_config.yml", 'r') as file:
                base_configs = yaml.safe_load(file)

            base_misc_info = base_configs["misc_info"]
            base_train_config= base_configs["training_config"]

            crits_dic = {}

            for k in range(num_repeat):

                seed = SEED + k

                base_model = model_callByName[base_misc_info["model_init"]](**base_misc_info["model_config"])
                
                train_X, test_X, recal_X, train_Y, test_Y, recal_Y = common_processor_UCI(x, y, recal_percent= 0., seed = seed)

                train_X, test_X = torch.Tensor(train_X), torch.Tensor(test_X)
                train_Y, test_Y = torch.Tensor(train_Y).to(torch.device("cuda")), torch.Tensor(test_Y).to(torch.device("cuda"))

                
                trainer(
                    seed = seed,
                    raw_train_X = train_X,
                    raw_train_Y = train_Y,
                    model = base_model,
                    training_config = base_train_config,
                    harvestor = None,          
                    misc_info = base_misc_info,
                    diff_trainingset = True
                )
                

            
                record = testPerform_muSigma(test_X, test_Y, model_name= modelname, model = base_model)

                if k == 0:
                    for key in record.keys():

                        crits_dic[modelname + "_"+key] = []

                for key in record.keys():

                    crits_dic[modelname + "_"+key].append(record[key])

            for key in crits_dic.keys():
                err_mu_dic[key] = (max(crits_dic[key]) + min(crits_dic[key]))/2
                err_std_dic[key] = (max(crits_dic[key]) - min(crits_dic[key]))/2


        if len(big_df) == 0:
            big_df["idxes"] = list(err_mu_dic.keys())

        big_df[dataname +"_mu"] = list(err_mu_dic.values())
        big_df[dataname + "_std"] = list(err_std_dic.values())
        
        
    df = pd.DataFrame.from_dict(big_df)  

    df.to_csv(os.getcwd()+"/Experiments/EXP1/record_bin/muSigma_benchmarks.csv",index=False)



