
import os
import yaml
from trainer import trainer, model_callByName, loss_callByName
from data_utils import get_uci_data, common_processor_UCI, seed_all
from TestPerform import testPerform_muSigma, testPerform_isotonic, testPerform_kernel, testPerform_projKernel
import torch
import pandas as pd
from src.evaluations import obs_vs_exp, mu_sig_toQuants
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import random_projection


if __name__ == "__main__":

    SEED = 1234
    num_repeat = 5
    big_df = {}

    for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:

        err_mu_dic = {}
        err_std_dic = {}

        dataset_path = os.getcwd()+"/Dataset/UCI_datasets"

        x, y = get_uci_data(data_name= dataname, dir_name= dataset_path)



        for modelname in ["pure_Kernel"]:


            print("model: "+ modelname +" on data: "+dataname)


            # train base model

            with open(os.getcwd()+"/Experiments/EXP1/config_bin/vanillaKernel_on_"+dataname+"_config.yml", 'r') as file:
                base_configs = yaml.safe_load(file)


            base_misc_info = base_configs["base_misc_info"]
            base_train_config= base_configs["base_train_config"]
            width = base_configs["wid"]

            crits_dic = {}

            for k in range(num_repeat):

                seed = SEED + k

                
                
                train_X, test_X, recal_X, train_Y, test_Y, recal_Y = common_processor_UCI(x, y, recal_percent= 0.5, seed = seed)

                train_X, test_X, recal_X = torch.Tensor(train_X), torch.Tensor(test_X), torch.Tensor(recal_X)
                train_Y, test_Y, recal_Y = torch.Tensor(train_Y).to(torch.device("cuda")), torch.Tensor(test_Y).to(torch.device("cuda")), torch.Tensor(recal_Y).to(torch.device("cuda"))


                


                record = testPerform_kernel(test_X, test_Y, recal_X, recal_Y, model_name= modelname, model = None, wid = width)


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

    df.to_csv(os.getcwd()+"/Experiments/EXP1/record_bin/pure_kernel.csv",index=False)



