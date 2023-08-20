from src.models import MC_dropnet, vanilla_predNet
from src.losses import mse_loss, rmse_loss, mean_std_norm_loss, MMD_Loss, mean_std_forEnsemble, BeyondPinball_muSigma, MACE_Loss
from src.kernel_methods import kernel_estimator, tau_to_quant_datasetCreate
from data_utils import seed_all, splitter, get_uci_data, common_processor_UCI
from trainer import trainer, empty_harvestor, model_callByName, loss_callByName
import copy
import torch
import os
import time
import numpy as np
import operator
import yaml




def grid_searcher_v2(
    dataset_name = "boston",
    dataset_path = os.getcwd()+"/Dataset/UCI_datasets",
    starting_seed = 1234,
    num_repeat = 5,
    model_name = "vanillaKernel",
    to_search = {
        "wid" : [5, 1, 5E-1, 1E-1, 5E-2, 1E-2] 
    },
    base_misc_info = {
        "input_x_shape": [455, 13],
        "input_y_shape": [455],
        "model_config": {
            "device": "cuda",
            "hidden_layers": [10, 5],
            "n_input": 13,
            "n_output": 1
        },
        "model_init": "vanillaPred",
        "save_path_and_name": None,
        "val_percentage": 0.1
    },
    base_train_config = {
        "Decay": 0.0001,
        "LR": 0.01,
        "N_Epoch": 200,
        "backdoor": None,
        "bat_size": 64,
        "early_stopping": True,
        "monitor_name": "mse",
        "patience": 20,
        "train_loss": "mse_loss",
        "val_loss_criterias":{
            "mse": "mse_loss",
            "rmse": "rmse_loss",
        },
        "validate_times": 20,
        "verbose": False
    },

    aux_config = {

    },
    base_harvestor = None,
    report_path = os.getcwd()+"/Experiments/EXP1/record_bin/",
    config_path = os.getcwd()+"/Experiments/EXP1/config_bin/"

):
    
    base_misc_info["model_init"] = model_name
    
    start_time = time.time()
    
    # for different model name, summarizer will take in different things
    summarizer = []
    support_summ = {}

    assert model_name in model_callByName.keys()

    for k in range(num_repeat):

        SEED = starting_seed + k

        sub_summarizer = {}
        aid_summarizer = {}

        seed_all(SEED)

        x, y = get_uci_data(data_name= dataset_name, dir_name= dataset_path)

        if model_name == "vanillaKernel":
            train_X, test_X, recal_X, train_Y, test_Y, recal_Y = common_processor_UCI(x, y, recal_percent= 0.5, seed = SEED)
        else:
            train_X, test_X, recal_X, train_Y, test_Y, recal_Y = common_processor_UCI(x, y, recal_percent= 0, seed = SEED)
        
        train_X = torch.Tensor(train_X)
        train_Y = torch.Tensor(train_Y).to(torch.device(base_misc_info["model_config"]["device"]))

        base_model = model_callByName[base_misc_info["model_init"]](**base_misc_info["model_config"])


        trainer(
            seed = SEED,
            raw_train_X = train_X,
            raw_train_Y = train_Y,
            model = base_model,
            training_config = base_train_config,
            harvestor = None,          
            misc_info = base_misc_info,
            diff_trainingset= True
        )

        # from now I guess we have finished training the first part of the model

        if model_name == "HNN_MMD": 
            # base model will be reused

            aux_misc_info = copy.deepcopy(base_misc_info)
            aux_train_config = copy.deepcopy(base_train_config)

            aux_train_config["train_loss"] = "MMD_Loss"

            aux_train_config["val_loss_criterias"] = {
                "MMD": "MMD_Loss",
                "MACE": "MACE_muSigma",
                "nll": "mean_std_norm_loss",
                "rmse": "rmse_loss"
            }
            aux_train_config["monitor_name"] = "MMD"

            harvestor = {
                "early_stopped": False,
                "early_stopping_epoch": 0,
                "monitor_name": "MACE",
                "monitor_vals": [],
                "training_losses": [],
                "val_MMD": [],
                "val_MACE": [],
                "val_nll": [],
                "val_rmse": []
            }



            assert "LR" in to_search.keys() and "bat_size" in to_search.keys()

            for LR in to_search["LR"]:
                for bat_size in to_search["bat_size"]:
                    aux_train_config["LR"] = LR
                    aux_train_config["bat_size"] = bat_size

                    empty_harvestor(harvestor)

                    model_reuse = copy.deepcopy(base_model)

                    trainer(
                        seed = SEED,
                        raw_train_X = train_X,
                        raw_train_Y = train_Y,
                        model = model_reuse,
                        training_config = aux_train_config,
                        harvestor = harvestor,          
                        misc_info = aux_misc_info
                    )

                
                    sub_summarizer[(LR, bat_size)] = np.mean(harvestor["monitor_vals"][-3:])
                    aid_summarizer[(LR, bat_size)] = {}

                    for key in aux_train_config["val_loss_criterias"].keys():
                        aid_summarizer[(LR, bat_size)]["val_"+key] = np.mean(harvestor["val_"+key][-3:])


            para_got = min(sub_summarizer.items(), key=operator.itemgetter(1))[0]
            summarizer.append(para_got)
            support_summ[para_got] = aid_summarizer[para_got]


        elif model_name == "vanillaMSQR": 

            # create a new dataset

            assert ("wid" in to_search) and ("LR" in to_search)

            for wid in to_search["wid"]:

                recal_data_amount = 500

                Z = train_X[:recal_data_amount].cuda()
                eps = (train_Y[:recal_data_amount] - base_model(Z).view(-1)).detach().cuda()

                reg_X, reg_Y = tau_to_quant_datasetCreate(Z, epsilon=eps, quants= np.linspace(0.01,0.99,20),wid = wid)

            

                aux_misc_info = copy.deepcopy(base_misc_info)

                aux_misc_info["input_x_shape"] = list(reg_X.shape)

                aux_misc_info["input_y_shape"] = list(reg_Y.shape)

                aux_misc_info["model_config"]["n_input"] = len(reg_X[0])

                aux_train_config = copy.deepcopy(base_train_config)

                aux_train_config["bat_size"] = 1024


                for LR in to_search["LR"]:
                
                    aux_train_config["LR"] = LR


                    model_quantpred = model_callByName[aux_misc_info["model_init"]](**aux_misc_info["model_config"])

                    trainer(
                        seed = SEED,
                        raw_train_X = reg_X,
                        raw_train_Y = reg_Y,
                        model = model_quantpred,
                        training_config = aux_train_config,
                        harvestor = None,          
                        misc_info = aux_misc_info
                    )

                    
                    # get the commonly used val data
                    split_percet = base_misc_info["val_percentage"]
    
                    N_val = int(split_percet*len(train_Y))

                    train_idx, val_idx = splitter(len(train_Y)-N_val, N_val, seed = SEED)

                    val_X, val_Y = train_X[val_idx], train_Y[val_idx]

                    quants = np.linspace(0.01,0.99,100)


                    quant_bed = torch.Tensor(quants).view(-1, 1).repeat(1, len(val_X)).view(-1).to(val_X.device)


                    val_X_stacked = val_X.repeat(len(quants), 1)
    
                    forReg_X = torch.cat([val_X_stacked, quant_bed.view(-1,1)], dim=1)

                    pred_eps = model_quantpred(forReg_X)

                    pred_eps = pred_eps.reshape(len(quants), len(val_Y))

                    val_mean = base_model(val_X)

                    pred_Y = pred_eps + val_mean.view(1,-1).repeat(len(pred_eps),1)

                    MACE_error = MACE_Loss(pred_Y,val_Y,q_list = np.linspace(0.01,0.99,100)).item()

                
                    sub_summarizer[(LR, wid)] = MACE_error
                    aid_summarizer[(LR, wid)] = MACE_error

        


            para_got = min(sub_summarizer.items(), key=operator.itemgetter(1))[0]
            summarizer.append(para_got)
            support_summ[para_got] = aid_summarizer[para_got]









        elif model_name == "vanillaKernel":

            recal_X = torch.Tensor(recal_X)
            recal_Y = torch.Tensor(recal_Y).to(torch.device(base_misc_info["model_config"]["device"]))

            # we need to get those val_x and val_y
            split_percet = base_misc_info["val_percentage"]
    
            N_val = int(split_percet*len(train_Y))

            train_idx, val_idx = splitter(len(train_Y)-N_val, N_val, seed = SEED)

            val_X, val_Y = train_X[val_idx], train_Y[val_idx]

            recal_mean = base_model(recal_X).view(-1)
            val_mean = base_model(val_X).view(-1)
            
            assert "wid" in to_search

            for wid in to_search["wid"]:

                eps_diffQuants = kernel_estimator(
                    test_Z = val_X.cuda(),
                    recal_Z = recal_X.cuda(),
                    recal_epsilon = torch.Tensor(recal_Y - recal_mean).cuda(),
                    quants = np.linspace(0.01,0.99,100),
                    wid= wid
                )

                y_diffQuants = eps_diffQuants + val_mean.view(1,-1).repeat(len(eps_diffQuants),1)
                MACE_error = MACE_Loss(y_diffQuants,val_Y,q_list = np.linspace(0.01,0.99,100)).item()
                

                sub_summarizer[(wid,)] = MACE_error
                aid_summarizer[(wid,)] = MACE_error

            para_got = min(sub_summarizer.items(), key=operator.itemgetter(1))[0]
            summarizer.append(para_got)
            support_summ[para_got] = aid_summarizer[para_got]
            

    
    if model_name == "HNN_MMD":

        choice_para = max(set(summarizer), key = summarizer.count)

        aid_dic = support_summ[choice_para]

        choice_LR, choice_bat_size = choice_para

    elif model_name == "vanillaKernel":

        choice_para = max(set(summarizer), key = summarizer.count)

        aid_dic = {"MACE": support_summ[choice_para]}

        choice_wid = choice_para[0]

    elif model_name == "vanillaMSQR":

        choice_para = max(set(summarizer), key = summarizer.count)

        aid_dic = {"MACE": support_summ[choice_para]}

        choice_LR, choice_wid = choice_para     






    finish_time = time.time()



        
    # now we need to write a report

    filename = dataset_name + "_report.txt"

    full_filename = report_path + filename

    if os.path.exists(full_filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    handle = open(full_filename, append_write)
    handle.write("\n\n")
    handle.write("# -------------------------------------------------")
    handle.write("\n\n")
    handle.write("On dataset: "+dataset_name +"\n")
    handle.write("We are training model: "+model_name+"\n")
    handle.write("With training X of size: ({0},{1})\n".format(train_X.shape[0],train_X.shape[1]))
    if model_name == "HNN_MMD":
        handle.write("After training an HNN model, we retrain it based on the MMD loss\n")
        handle.write("And grid searching for the best LR and bat_size\n")
        handle.write("After {0:.2f} hours of training\n".format((finish_time - start_time)/3600))
        handle.write("We get a few ideal choices for tuple (LR, bat_size)\n")
        for tup in summarizer:
            handle.write("\t ({0}, {1})\n".format(tup[0], tup[1]))
        handle.write("we finally choose ({0}, {1}) as the best hyperparameters\n".format(choice_LR, choice_bat_size))

    
    elif model_name == "vanillaKernel":
        handle.write("After training a vanilla prediction model\n")
        handle.write("And grid searching for the best kernel width\n")
        handle.write("After {0:.2f} hours of training\n".format((finish_time - start_time)/3600))
        handle.write("We get a few ideal choices for tuple the width\n")
        for tup in summarizer:
            handle.write("\t {0}\n".format(tup[0]))
        handle.write("we finally choose {0} as the best hyperparameters\n".format(tup[0]))

    
    elif model_name == "vanillaMSQR":
        handle.write("After training a vanilla prediction model\n")
        handle.write("We search for the best width and LR for the MSQR algorithm\n")
        handle.write("After {0:.2f} hours of training\n".format((finish_time - start_time)/3600))
        handle.write("We get a few ideal choices for tuple (LR, wid)\n")
        for tup in summarizer:
            handle.write("\t ({0}, {1})\n".format(tup[0], tup[1]))
        handle.write("we finally choose ({0}, {1}) as the best hyperparameters\n".format(choice_LR, choice_wid))


    
    handle.write("with corresponding evaluations:\n")
    for key in aid_dic.keys():
        handle.write("\t"+key+": "+str(aid_dic[key])+"\n")
    handle.write("All configs are recorded into yaml files in the config directory\n")
    handle.write("\n\n")
    handle.write("# -------------------------------------------------")
    handle.write("\n\n")
    handle.close()


    # dump config records
    config_name = model_name + "_on_" + dataset_name + "_config.yml"



    if model_name == "HNN_MMD":

        empty_harvestor(harvestor)

        aux_train_config["LR"] = choice_LR
        aux_train_config["bat_size"] = choice_bat_size


        with open(config_path+config_name, 'w') as config_handle:

            yaml.dump(
                {
                    "base_misc_info": base_misc_info,
                    "base_train_config": base_train_config,
                    "aux_misc_info": aux_misc_info,
                    "aux_train_config": aux_train_config,
                    "harvestor": harvestor
                }, 
                config_handle, 
                default_flow_style=False)
            
    elif model_name == "vanillaKernel":

        with open(config_path+config_name, 'w') as config_handle:

            yaml.dump(
                {
                    "base_misc_info": base_misc_info,
                    "base_train_config": base_train_config,
                    "wid": choice_wid
                }, 
                config_handle, 
                default_flow_style=False)
            
            
    elif model_name == "vanillaMSQR":

        with open(config_path+config_name, 'w') as config_handle:

            yaml.dump(
                {
                    "base_misc_info": base_misc_info,
                    "base_train_config": base_train_config,
                    "aux_misc_info": aux_misc_info,
                    "aux_train_config": aux_train_config,
                    "wid": choice_wid
                }, 
                config_handle, 
                default_flow_style=False)





if __name__ == "__main__":
    '''
    with open(os.getcwd()+"/Experiments/EXP1/config_bin/HNN_on_energy_config.yml", 'r') as file:
        base_configs = yaml.safe_load(file)

    
    grid_searcher_v2(
        dataset_name= "energy",
        num_repeat=2,
        model_name = "HNN_MMD",
        to_search = {
            "LR" : [5E-3, 1E-3],
            "bat_size": [600]
        },
        base_misc_info = base_configs["misc_info"],
        base_train_config= base_configs["training_config"],
        aux_config = {},
    )
    '''


    '''
    with open(os.getcwd()+"/Experiments/EXP1/config_bin/vanillaPred_on_energy_config.yml", 'r') as file:
        base_configs = yaml.safe_load(file)

    grid_searcher_v2(
        dataset_name= "energy",
        num_repeat=2,
        model_name = "vanillaKernel",
        to_search = {"wid" : [5E-1, 1E-1]},
        base_misc_info = base_configs["misc_info"],
        base_train_config= base_configs["training_config"],
        aux_config = {},
    )
    '''


    '''
    with open(os.getcwd()+"/Experiments/EXP1/config_bin/vanillaPred_on_energy_config.yml", 'r') as file:
        base_configs = yaml.safe_load(file)

    grid_searcher_v2(
        dataset_name= "energy",
        num_repeat=2,
        model_name = "vanillaMSQR",
        to_search = {
            "wid" : [5E-1, 1E-1],
            "LR": [5E-3]
            },
        base_misc_info = base_configs["misc_info"],
        base_train_config= base_configs["training_config"],
        aux_config = {},
    )
    '''

    pass


