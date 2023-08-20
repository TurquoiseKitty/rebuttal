import pandas as pd
from data_utils import splitter, seed_all, OnlineNews, normalize
import numpy as np
import torch
from src.models import vanilla_predNet, MC_dropnet, Deep_Ensemble
from src.losses import *
from Experiments.EXP1.TestPerform import testPerform_muSigma
from Experiments.EXP1.TestPerform import testPerform_projKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn import random_projection
import os




if __name__ == "__main__":

    x, y = OnlineNews()
    big_df = {}

    x_normed, normalizer = normalize(x)
    x = x_normed


    for n_comp in [1,5,10,15]:

        

        

        print("try n component = ", n_comp)

        comp_dic = {}


        for repeat in range(5):


            SEED = 5678 + repeat

            seed_all(SEED)

            N_train = int(len(x) * 0.9)
            N_test = int(len(x) * 0.1)

            tr_idx = np.arange(len(x))[:N_train]

            te_idx = np.arange(len(x))[N_train:N_train+N_test]


            test_X, test_Y = x[te_idx], y[te_idx]



            x_reshaped, y_reshaped = x[tr_idx], y[tr_idx]


            N_model_train = int(len(y_reshaped) * 0.9)
            N_recalibration = int(len(y_reshaped) * 0.1)

            tr_new_idx, recal_idx = splitter(N_model_train, N_recalibration, seed = SEED)


            recal_X = x_reshaped[recal_idx]
            recal_Y = y_reshaped[recal_idx]



            x_remain, y_remain = x_reshaped[tr_new_idx], y_reshaped[tr_new_idx]


            split = 0.8
            train_idx, val_idx = splitter(int(split * len(y_remain)), len(y_remain) - int(split * len(y_remain)), seed = SEED)


            train_X, train_Y = x_remain[train_idx], y_remain[train_idx]
            val_X, val_Y = x_remain[val_idx], y_remain[val_idx]

            train_X = torch.Tensor(train_X)
            train_Y = torch.Tensor(train_Y).view(-1).cuda()
            val_X = torch.Tensor(val_X)
            val_Y = torch.Tensor(val_Y).view(-1).cuda()
            test_X = torch.Tensor(test_X)
            test_Y = torch.Tensor(test_Y).view(-1).cuda()

            recal_X = torch.Tensor(recal_X)
            recal_Y = torch.Tensor(recal_Y).view(-1).cuda()


            n_feature = train_X.shape[1]

            hidden = [100, 50]
            epochs = 200

            pred_model = vanilla_predNet(
                n_input = n_feature,
                hidden_layers = hidden
            )


            pred_model.train(
                train_X, train_Y, val_X, val_Y,
                bat_size = 256,
                LR = 5E-3,

                N_Epoch = epochs,
                validate_times = 20,
                verbose = False,
                train_loss = mse_loss,
                val_loss_criterias = {
                    "mse": mse_loss,
                    "rmse": rmse_loss,
                },
                early_stopping = True,
                patience = 20,
                monitor_name = "rmse"
            )




            # try different widths

            if n_comp < x.shape[1]:

                transformer = random_projection.GaussianRandomProjection(n_components = n_comp)
                reformer = lambda x : torch.Tensor(transformer.fit_transform(x.cpu().numpy()))

            else:

                reformer = lambda x : x

            record = {}


            for width in [1,5,13,26,60]:

                raw_record = testPerform_projKernel(
                    test_X, test_Y, recal_X, recal_Y, 
                    model_name = "vanillaKernel_RandomProj", model= pred_model, reformer= reformer, wid = width)
                
                if len(record) == 0:

                    for key in raw_record.keys():

                        record[key] = 1E6

                for key in raw_record.keys():

                    record[key] = min(record[key], raw_record[key])


            if repeat == 0:

                for key in record.keys():

                    comp_dic[key] = []

                
            for key in record.keys():

                comp_dic[key].append(record[key])


        temp_dic_mu = {}
        temp_dic_std = {}

        for key in comp_dic.keys():

            temp_dic_mu[key] = np.mean(comp_dic[key])
            temp_dic_std[key] = np.std(comp_dic[key]) / np.sqrt(len(comp_dic[key]))



        if len(big_df) == 0:
            big_df["idxes"] = list(comp_dic.keys())

        
        big_df["comp_"+str(int(n_comp)) +"_mu"] = list(temp_dic_mu.values())
        big_df["comp_"+str(int(n_comp)) +"_std"] = list(temp_dic_std.values())


    df = pd.DataFrame.from_dict(big_df)  

    df.to_csv(os.getcwd()+"/Experiments/EXP4/record_bin/OnlineNews_vanillaKernel_proj.csv",index=False)




















