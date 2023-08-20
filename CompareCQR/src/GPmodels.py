# code from https://docs.gpytorch.ai/en/latest/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html

import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from .losses import mse_loss, rmse_loss, mean_std_norm_loss


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class oneLayer_DeepGP(DeepGP):

    

    def __init__(self, n_input, hidden_layers, device = torch.device('cuda'), **kwargs):

        in_dim = n_input
        hidden_dim = hidden_layers[0]

        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=in_dim,
            output_dims=hidden_dim,
            mean_type='linear',
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

        self.device = device

        self.to(device)

    def forward(self, inputs):

        assert isinstance(inputs, torch.Tensor)
        assert len(inputs.shape) == 2

        inputs = inputs.to(self.device)

        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, inputs, bat_size = 128):

        assert isinstance(inputs, torch.Tensor)
        assert len(inputs.shape) == 2

        val_set = TensorDataset(inputs)
        val_loader = DataLoader(val_set, batch_size=bat_size, shuffle=False)

        with torch.no_grad():
            mus = []
            variances = []
            
            for x_batch in val_loader:

                x_batch = x_batch[0].to(self.device)


                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)


        return torch.stack((torch.cat(mus, dim=-1).mean(dim = 0), torch.sqrt(torch.cat(variances, dim=-1).mean(dim = 0))), dim = 1)
    
    def train(self, 
              X_train, Y_train, X_val, Y_val,
              bat_size = 128,
              LR = 1E-2,
              Decay = 1E-4,
              N_Epoch = 100,
              num_samples = 30,
              validate_times = 20,
              verbose = True,
              val_loss_criterias = {
                  "nll" : mean_std_norm_loss,
                  "rmse": rmse_loss
              },
              early_stopping = True,
              patience = 10,
              monitor_name = "nll",
              harvestor = None,
              **kwargs
              ):

        optimizer = torch.optim.Adam([
            {'params': self.parameters()},
        ], lr=LR, weight_decay=Decay)


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
        training_loader = DataLoader(training_set, batch_size=bat_size, shuffle=True)


        mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, X_train.shape[-2]))

        PREV_loss = 1E5

        if early_stopping:
            patience_count = 0

        for epoch in range(N_Epoch):
            for i_bat, (X_bat, Y_bat) in enumerate(training_loader):

                with gpytorch.settings.num_likelihood_samples(num_samples):
                    optimizer.zero_grad()
                    output = self(X_bat)
                    loss = -mll(output, Y_bat)
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

                    