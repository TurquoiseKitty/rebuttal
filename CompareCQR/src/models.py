# refer to https://github.com/JavierAntoran/Bayesian-Neural-Networks

# MC dropout for heteroskedastic network.


from .DEFAULTS import DEFAULT_layers
import torch
import torch.nn as nn
from .losses import mse_loss, rmse_loss, mean_std_norm_loss
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader



class raw_net(nn.Module):

    def __init__(
            self,
            n_input,
            hidden_layers,
            drop_rate = 0.5,
            device = torch.device('cuda'),
            **kwargs
    ):
        super(raw_net, self).__init__()

        self.n_input = n_input
        self.hidden_layers = hidden_layers
        self.device = device
        self.drop_rate = drop_rate

        model_seq = []

        prev_dim = n_input
        for dimi in hidden_layers:

            model_seq.append(nn.Linear(prev_dim, dimi))
            model_seq.append(nn.LeakyReLU(0.2, inplace=True))

            if self.drop_rate > 1E-4:
                model_seq.append(nn.Dropout(drop_rate))

            prev_dim = dimi

        self.raw_model = nn.ModuleList(model_seq).to(self.device)


    def forward(
            self, x, **kwargs
    ):
        raise NotImplementedError
    

    def predict(
            self, x, **kwargs
    ):
        raise NotImplementedError
    
            
    def train(self,
              X_train, Y_train, X_val, Y_val,
              bat_size = 128,
              LR = 1E-2,
              Decay = 1E-4,
              N_Epoch = 300,
              validate_times = 20,
              verbose = True,
              train_loss = mean_std_norm_loss,
              val_loss_criterias = {
                  "nll" : mean_std_norm_loss,
                  "rmse": rmse_loss
              },
              early_stopping = True,
              patience = 10,
              monitor_name = "nll",
              backdoor = None,
              harvestor = None,
              **kwargs
            ):
        
        optimizer = optim.Adam(self.parameters(), lr = LR, weight_decay=Decay)


        if not harvestor:

            harvestor = {
                "training_losses": []
            }
            if early_stopping:
                harvestor["early_stopped"] = False
                harvestor["early_stopping_epoch"] = 0
                harvestor["monitor_name"] = monitor_name
                harvestor["monitor_vals"] = []

            for key in val_loss_criterias.keys():
                harvestor["val_"+key] = []

        else:

            # we are assuming that a harvestor is carefully written and carefully inserted

            assert len(harvestor["training_losses"]) == 0

            if early_stopping:

                assert "early_stopped" in harvestor.keys() and not harvestor["early_stopped"]
                assert harvestor["early_stopping_epoch"] == 0
                assert len(harvestor["monitor_vals"]) == 0

            for key in val_loss_criterias.keys():
                assert len(harvestor["val_"+key]) == 0


            


        if isinstance(X_train, np.ndarray):

            X_train, Y_train, X_val, Y_val = map(torch.Tensor, [X_train, Y_train, X_val, Y_val])
        

        training_set = TensorDataset(X_train, Y_train)

        if backdoor and backdoor == "MMD_LocalTrain":
            training_loader = DataLoader(training_set, batch_size=bat_size, shuffle=False)
        else:
            training_loader = DataLoader(training_set, batch_size=bat_size, shuffle=True)

        PREV_loss = 1E5

        if early_stopping:
            patience_count = 0

        for epoch in range(N_Epoch):
            for i_bat, (X_bat, Y_bat) in enumerate(training_loader):

                optimizer.zero_grad()

                loss = train_loss(self.forward(X_bat), Y_bat)

                loss.backward()

                optimizer.step()


            # we always want to validate
            harvestor["training_losses"].append(loss.item())

            val_output = self.predict(X_val)

            if early_stopping:
                patience_val_loss = val_loss_criterias[monitor_name](val_output, Y_val).item()

                harvestor["monitor_vals"].append(patience_val_loss)

                if patience_val_loss > PREV_loss:
                    patience_count += 1
                
                PREV_loss = patience_val_loss


            

            
            if early_stopping and patience_count >= patience:

                if verbose:

                    print("Early Stopped at Epoch ", epoch)
                
                harvestor["early_stopped"] = True
                harvestor["early_stopping_epoch"] = epoch


                break

            if epoch % int(N_Epoch / validate_times) == 0:

                
                if verbose:
                    print("epoch ", epoch)
                for name in val_loss_criterias.keys():

                    val_loss = val_loss_criterias[name](val_output, Y_val).item()

                    harvestor["val_"+name].append(val_loss)

                    if verbose:
                        print("     loss: {0}, {1}".format(name, val_loss))


class vanilla_predNet(raw_net):

    def __init__(
            self,
            n_input,
            hidden_layers,
            n_output = 1,
            device = torch.device('cuda'),
            **kwargs
    ): 

        super(vanilla_predNet, self).__init__(
            n_input= n_input,
            hidden_layers= hidden_layers,
            drop_rate= 0,
            device= device
        )

        self.n_output = n_output

        if len(hidden_layers) > 0:

            dim_before = hidden_layers[-1]
        else:
            dim_before = n_input

        self.tail = nn.Linear(dim_before, n_output).to(self.device)


    def forward(self, x: torch.Tensor):

        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        x = x.to(self.device)

        for m in self.raw_model:

            x = m(x)

        x = self.tail(x)

        return x
    

    def predict(self, 
                x: torch.Tensor,
                bat_size = 128,
                ):
        
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        # sometimes validate set might get too big

        val_set = TensorDataset(x)
        val_loader = DataLoader(val_set, batch_size=bat_size, shuffle=False)

        with torch.no_grad():
            mus = []

            for x_batch in val_loader:


                x = x_batch[0].to(self.device)

                out = self(x)                

                mus.append(out[:,0])


        return torch.cat(mus, dim=-1)


class quantile_predNet(vanilla_predNet):

    
    def __init__(
            self,
            n_input,
            hidden_layers,
            n_output,
            device = torch.device('cuda'),
            **kwargs
    ): 

        super(quantile_predNet, self).__init__(
            n_input= n_input,
            hidden_layers= hidden_layers,
            n_output= n_output,
            device= device
        )


    def predict(self, 
                x: torch.Tensor,
                bat_size = 128,
                ):
        
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        # sometimes validate set might get too big

        val_set = TensorDataset(x)
        val_loader = DataLoader(val_set, batch_size=bat_size, shuffle=False)

        with torch.no_grad():
            outs = []

            for x_batch in val_loader:


                x = x_batch[0].to(self.device)

                out = self(x)                

                outs.append(out)


        return torch.cat(outs, dim=0)







class MC_dropnet(raw_net):

    def __init__(
            self,
            n_input,
            hidden_layers,
            n_output = 2,
            drop_rate = 0.1,
            device = torch.device('cuda'),
            **kwargs
    ): 
        
        # we only implement heteroskedastic setting
        assert n_output == 2

        super(MC_dropnet, self).__init__(
            n_input= n_input,
            hidden_layers= hidden_layers,
            drop_rate= drop_rate,
            device= device
        )

        self.n_output = n_output

        if len(hidden_layers) > 0:

            dim_before = hidden_layers[-1]
        else:
            dim_before = n_input

        self.tail = nn.Linear(dim_before, n_output).to(self.device)


    def forward(self, x: torch.Tensor):

        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        x = x.to(self.device)

        for m in self.raw_model:

            x = m(x)

        x = self.tail(x)

        mu = x[:, :1]
        raw_sigma = x[:, 1:]
        sigma = nn.Softplus()(raw_sigma)


        return torch.cat((mu, sigma), axis = 1)
    

    def predict(self, 
                x: torch.Tensor,
                bat_size = 128,
                trial = 100):
        
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        # sometimes validate set might get too big

        val_set = TensorDataset(x)
        val_loader = DataLoader(val_set, batch_size=bat_size, shuffle=False)

        with torch.no_grad():
            mus = []
            sigs = []

            for x_batch in val_loader:


                x = x_batch[0].to(self.device)

                if self.drop_rate > 1E-4:
                    samples = []
                    noises = []

                    for i in range(trial):
                        preds = self(x)
                        samples.append(preds[:, 0])
                        noises.append(preds[:, 1])

                    samples = torch.stack(samples, dim = 0)
                    noises = torch.stack(noises, dim = 0)
                    assert samples.shape == (trial, len(x))

                    mean_preds = samples.mean(dim = 0)
                    aleatoric = (noises**2).mean(dim = 0)**0.5

                    epistemic = samples.var(dim = 0)**0.5
                    total_unc = (aleatoric**2 + epistemic**2)**0.5

                    out = torch.stack((mean_preds, aleatoric), dim = 1)



                else:
                    out = self(x)

                

                mus.append(out[:,0])
                sigs.append(out[:,1])

        return  torch.stack((torch.cat(mus, dim=-1), torch.cat(sigs, dim=-1)), dim = 1)


class Deep_Ensemble(raw_net):

    def __init__(
            self,
            n_input,
            hidden_layers,
            n_output = 2,
            n_models = 5,
            device = torch.device('cuda'),
            **kwargs
    ):
        
        assert n_output == 2


        super(Deep_Ensemble, self).__init__(
            n_input= 2,
            hidden_layers= hidden_layers,
            drop_rate= 0,
            device= device
        )
        
        ensembles_list = []
        self.n_models = n_models
        self.device = device

        for i in range(n_models):

            ensembles_list.append(
                MC_dropnet(
                n_input= n_input,
                hidden_layers= hidden_layers,
                n_output= n_output,
                drop_rate= 0.,
                device= device
                )
            )
        
        self.ensembles = nn.ModuleList(ensembles_list)

    def forward(self, x: torch.Tensor):

        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2

        x = x.to(self.device)

        fin_output = []

        for mc_model in self.ensembles:

            out = mc_model(x)

            fin_output.append(out)

        
        return torch.cat(fin_output, dim= 0)


    def predict(self, x: torch.Tensor):
        
        raw_out = self(x)


        assert raw_out.shape == (self.n_models * len(x), 2)

        splitted = torch.stack(torch.split(raw_out, len(x)), dim = 0)

        samples = splitted[:, :, 0]
        noises = splitted[:, :, 1]

        mean_preds = samples.mean(dim = 0)
        aleatoric = (noises**2).mean(dim = 0)**0.5

        return torch.stack((mean_preds, aleatoric), dim = 1)






      

    
        



