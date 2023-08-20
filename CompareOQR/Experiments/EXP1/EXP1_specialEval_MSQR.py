
import os
import yaml
from trainer import trainer, model_callByName, loss_callByName
from data_utils import get_uci_data, common_processor_UCI, seed_all
from TestPerform import testPerform_muSigma, testPerform_quants
import torch
import pandas as pd
from src.evaluations import obs_vs_exp, mu_sig_toQuants
import numpy as np
from src.kernel_methods import tau_to_quant_datasetCreate


if __name__ == "__main__":

    SEED = 1234
    num_repeat = 5
    big_df = {}

    for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:

        

        err_mu_dic = {}
        err_std_dic = {}

        dataset_path = os.getcwd()+"/Dataset/UCI_datasets"

        x, y = get_uci_data(data_name= dataname, dir_name= dataset_path)

        modelname = "vanillaMSQR"

        print("model: "+ modelname +" on data: "+dataname)

        # train base model

        with open(os.getcwd()+"/Experiments/EXP1/config_bin/" +modelname +"_on_"+dataname+"_config.yml", 'r') as file:
            base_configs = yaml.safe_load(file)


        base_misc_info = base_configs["base_misc_info"]

        base_train_config= base_configs["base_train_config"]

        wid = base_configs["wid"]

        aux_misc_info = base_configs["aux_misc_info"]
        aux_train_config= base_configs["aux_train_config"]

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
    
            exp = np.linspace(0.01, 0.99, 100)

            recal_data_amount = 500

            Z = train_X[:recal_data_amount].cuda()
            eps = (train_Y[:recal_data_amount] - base_model(Z).view(-1)).detach().cuda()


            reg_X, reg_Y = tau_to_quant_datasetCreate(Z, epsilon=eps, quants= np.linspace(0.01,0.99,20),wid = wid)



            model_quantpred = model_callByName[aux_misc_info["model_init"]](**aux_misc_info["model_config"])

            trainer(
                seed = seed,
                raw_train_X = reg_X,
                raw_train_Y = reg_Y,
                model = model_quantpred,
                training_config = aux_train_config,
                harvestor = None,          
                misc_info = aux_misc_info,
                diff_trainingset = True
            )

            quants = exp


            quant_bed = torch.Tensor(quants).view(-1, 1).repeat(1, len(test_X)).view(-1).to(test_X.device)


            test_X_stacked = test_X.repeat(len(quants), 1)

            forReg_X = torch.cat([test_X_stacked, quant_bed.view(-1,1)], dim=1)

            pred_eps = model_quantpred(forReg_X)

            pred_eps = pred_eps.reshape(len(quants), len(test_Y))

            test_mean = base_model(test_X)

            pred_Y = pred_eps + test_mean.view(1,-1).repeat(len(pred_eps),1)

            record = testPerform_quants(pred_Y, test_Y, q_list= quants)

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

    df.to_csv(os.getcwd()+"/Experiments/EXP1/record_bin/MSQR_benchmarks.csv",index=False)



