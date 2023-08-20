import pandas as pd
from data_utils import splitter, seed_all, normalize
import numpy as np
import torch
from src.models import vanilla_predNet, MC_dropnet, Deep_Ensemble
from src.losses import *
from Experiments.EXP1.TestPerform import testPerform_muSigma
from sklearn.ensemble import RandomForestRegressor
from sklearn import random_projection
import os




if __name__ == "__main__":

    raw_df = pd.read_csv("Dataset/BlogFeedback/blogData_train.csv", header = None)

    raw_x = raw_df.iloc[:, :280].to_numpy()

    raw_y = raw_df.iloc[:, 280].to_numpy()

    # we only select those y >= 1

    raw_x = raw_x[raw_y >= 1]
    raw_y = raw_y[raw_y >= 1]
    raw_y = np.clip(np.log(raw_y), 0, 7)

    big_df = {}


    for n_comp in [4, 10, 20, 50, 100, 150]:

        # doing random projection before the training
        comp_dic = {}

        transformer = random_projection.GaussianRandomProjection(n_components = n_comp)
        reformer = lambda x : transformer.fit_transform(x)

        x = reformer(raw_x)
        y = raw_y

        print("try n component = ", n_comp)


        for repeat in range(5):


            SEED = 1234 + repeat

            seed_all(SEED)

            x_normed, x_normalizer = normalize(x)

            x = x_normed

            N_train = int(len(x) * 0.9)
            N_test = int(len(x) * 0.1)

            tr_idx = np.arange(len(x))[:N_train]

            te_idx = np.arange(len(x))[N_train:N_train+N_test]


            test_X, test_Y = x[te_idx], y[te_idx]



            x_reshaped, y_reshaped = x[tr_idx], y[tr_idx]


            N_model_train = int(len(y_reshaped) * 0.7)
            N_recalibration = int(len(y_reshaped) * 0.3)

            tr_new_idx, recal_idx = splitter(N_model_train, N_recalibration, seed = SEED)


            recal_X = x_reshaped[recal_idx]
            recal_Y = y_reshaped[recal_idx]



            x_remain, y_remain = x_reshaped[tr_new_idx], y_reshaped[tr_new_idx]


            split = 0.8
            train_idx, val_idx = splitter(int(split * len(y_remain)), len(y_remain) - int(split * len(y_remain)), seed = SEED)


            train_X, train_Y = x_remain[train_idx], y_remain[train_idx]
            val_X, val_Y = x_remain[val_idx], y_remain[val_idx]

            n_feature = x.shape[1]

            epochs = 300


            train_X = torch.Tensor(train_X)
            train_Y = torch.Tensor(train_Y).view(-1).cuda()
            val_X = torch.Tensor(val_X)
            val_Y = torch.Tensor(val_Y).view(-1).cuda()
            test_X = torch.Tensor(test_X)
            test_Y = torch.Tensor(test_Y).view(-1).cuda()

            recal_X = torch.Tensor(recal_X)
            recal_Y = torch.Tensor(recal_Y).view(-1).cuda()


            hidden = [np.clip(n_feature, 10, 100), 10]

            deepEnsemble_model = Deep_Ensemble(
                n_input = n_feature,
                hidden_layers = hidden
            )


            deepEnsemble_model.train(
                train_X, train_Y, val_X, val_Y,
                bat_size = 64,
                LR = 5E-3,

                N_Epoch = epochs,
                validate_times = 20,
                verbose = True,
                train_loss = mean_std_forEnsemble,
                val_loss_criterias = {
                    "nll" : mean_std_forEnsemble,
                    "rmse": rmse_loss,
                    "MACE": MACE_muSigma
                },
                early_stopping = True,
                patience = 20,
                monitor_name = "rmse"
            )

            record = testPerform_muSigma(test_X, test_Y, model_name= "DeepEnsemble", model = deepEnsemble_model)




            if repeat == 0:

                for key in record.keys():

                    comp_dic[key] = []

                
            for key in record.keys():

                comp_dic[key].append(record[key])


        temp_dic_mu = {}
        temp_dic_std = {}

        for key in comp_dic.keys():

            temp_dic_mu[key] = (max(comp_dic[key]) + min(comp_dic[key]))/2
            temp_dic_std[key] = (max(comp_dic[key]) - min(comp_dic[key]))/2



        if len(big_df) == 0:
            big_df["idxes"] = list(comp_dic.keys())

        
        big_df["comp_"+str(int(n_comp)) +"_mu"] = list(temp_dic_mu.values())
        big_df["comp_"+str(int(n_comp)) +"_std"] = list(temp_dic_std.values())


    df = pd.DataFrame.from_dict(big_df)  

    df.to_csv(os.getcwd()+"/Experiments/EXP3/record_bin/DE_proj_benchmarks.csv",index=False)

        



