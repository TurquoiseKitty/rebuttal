import torch
from Experiments.EXP1.trainer import model_callByName, loss_callByName
from src.kernel_methods import kernel_estimator
import numpy as np

def testPerform_muSigma(
    test_X,
    test_Y,
    model_name,
    model,
    val_criterias = [
        "rmse_loss", "mean_std_norm_loss", "MACE_muSigma", "AGCE_muSigma", "CheckScore_muSigma"
    ]       
):
    assert model_name in ["HNN", "MC_drop", "DeepEnsemble", "HNN_BeyondPinball", "GPmodel", "HNN_MMD"]

    # return a single value for each criteria

    ret = {}

    y_out = model.predict(test_X)

    for key in val_criterias:

        real_loss = loss_callByName[key]

        

        real_err = real_loss(y_out, test_Y)

        if isinstance(real_err, torch.Tensor):

            real_err = real_err.item()

        ret[key] = real_err

    return ret


def testPerform_isotonic(
    test_X,
    test_Y,
    model_name,
    model,
    recal_model,
    val_criterias = [
        "MACE_muSigma", "AGCE_muSigma", "CheckScore_muSigma"
    ]       
):
    assert model_name in ["HNN", "MC_drop", "DeepEnsemble", "HNN_BeyondPinball"]

    # return a single value for each criteria

    ret = {}

    y_out = model.predict(test_X)

    for key in val_criterias:

        real_loss = loss_callByName[key]

        

        real_err = real_loss(y_out, test_Y, recal = True, recal_model = recal_model)

        if isinstance(real_err, torch.Tensor):

            real_err = real_err.item()

        ret[key] = real_err

    return ret



def testPerform_quants(
    y_diffQuants, 
    test_Y, 
    q_list = np.linspace(0.01,0.99,100),
    val_criterias = [
        "MACE_Loss", "AGCE_Loss", "CheckScore"
    ]
):
    
    ret = {}

    for key in val_criterias:

        real_loss = loss_callByName[key]

        real_err = real_loss(y_diffQuants, test_Y, q_list = q_list).item()

        if isinstance(real_err, torch.Tensor):

            real_err = real_err.item()

        ret[key] = real_err

    return ret

    


def testPerform_kernel(
    test_X,
    test_Y,
    recal_X,
    recal_Y,
    model_name,
    model,
    val_criterias = [
        "MACE_Loss", "AGCE_Loss", "CheckScore"
    ],
    wid = 10   
):
    assert model_name in ["RFKernel", "vanillaKernel",  "vanillaKernel_RandomProj", "vanillaKernel_CovSelect", "pure_Kernel"]

    # return a single value for each criteria

    ret = {}

    if model_name == "RFKernel":

        recal_mean = torch.Tensor(model.predict(recal_X.cpu().numpy())).cuda()
        test_mean = torch.Tensor(model.predict(test_X.cpu().numpy())).cuda()

    elif model_name == "pure_Kernel":

        recal_mean = torch.zeros(len(recal_X)).cuda()
        test_mean = torch.zeros(len(test_X)).cuda()

    else:

        recal_mean = model.predict(recal_X).view(-1)
        test_mean = model.predict(test_X).view(-1)


    eps_diffQuants = kernel_estimator(
        test_Z = test_X.cuda(),
        recal_Z = recal_X.cuda(),
        recal_epsilon = torch.Tensor(recal_Y - recal_mean).cuda(),
        quants = np.linspace(0.01,0.99,100),
        wid= wid
    )

    y_diffQuants = eps_diffQuants + test_mean.view(1,-1).repeat(len(eps_diffQuants),1)

    for key in val_criterias:

        real_loss = loss_callByName[key]

        real_err = real_loss(y_diffQuants, test_Y, q_list = np.linspace(0.01,0.99,100)).item()

        if isinstance(real_err, torch.Tensor):

            real_err = real_err.item()

        ret[key] = real_err

    return ret


def testPerform_projKernel(
    test_X,
    test_Y,
    recal_X,
    recal_Y,
    
    model_name,
    model,

    reformer,

    val_criterias = [
        "MACE_Loss", "AGCE_Loss", "CheckScore"
    ],
    wid = 10   
):


    assert model_name in ["RFKernel_RandomProj", "vanillaKernel_RandomProj", "vanillaKernel_CovSelect"]

    # return a single value for each criteria

    ret = {}

    if model_name in ["RFKernel_RandomProj"]:
        
        recal_mean = torch.Tensor(model.predict(recal_X.cpu().numpy())).cuda()
        test_mean = torch.Tensor(model.predict(test_X.cpu().numpy())).cuda()


    elif model_name in ["vanillaKernel_RandomProj", "vanillaKernel_CovSelect"]:

        

        recal_mean = model.predict(recal_X).view(-1)
        # recal_mean = model(recal_X).view(-1)

        test_mean = model.predict(test_X).view(-1)

    test_Z =  reformer(test_X)

    recal_Z = reformer(recal_X)

    

    
    eps_diffQuants = kernel_estimator(
        test_Z = test_Z.cuda(),
        recal_Z = recal_Z.cuda(),
        recal_epsilon = torch.Tensor(recal_Y - recal_mean).cuda(),
        quants = np.linspace(0.01,0.99,100),
        wid= wid
    )

    y_diffQuants = eps_diffQuants + test_mean.view(1,-1).repeat(len(eps_diffQuants),1)

    for key in val_criterias:

        real_loss = loss_callByName[key]

        real_err = real_loss(y_diffQuants, test_Y, q_list = np.linspace(0.01,0.99,100)).item()

        if isinstance(real_err, torch.Tensor):

            real_err = real_err.item()

        ret[key] = real_err

    return ret

