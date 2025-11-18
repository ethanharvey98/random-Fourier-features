# PyTorch
import torch
# Importing our custom module(s)
import utils

class RandomFourierFeatures(torch.nn.Module):
    def __init__(self, in_features, rank=1024, lengthscale=20.0, outputscale=1.0, learnable_lengthscale=False, learnable_outputscale=False):
        super().__init__()
        
        self.rank = rank
        self.register_buffer("feature_weight", torch.randn(self.rank, in_features))
        self.register_buffer("feature_bias", 2 * torch.pi * torch.rand(self.rank))
                
        if learnable_lengthscale:
            self.raw_lengthscale = torch.nn.Parameter(utils.inv_softplus(torch.tensor(lengthscale, dtype=torch.float32)))
        else:
            self.register_buffer("raw_lengthscale", utils.inv_softplus(torch.tensor(lengthscale, dtype=torch.float32)))
        
        if learnable_outputscale:
            self.raw_outputscale = torch.nn.Parameter(utils.inv_softplus(torch.tensor(outputscale, dtype=torch.float32)))
        else:
            self.register_buffer("raw_outputscale", utils.inv_softplus(torch.tensor(outputscale, dtype=torch.float32)))
                    
    def featurize(self, x):
        return self.outputscale * (2/self.rank)**0.5 * torch.cos(torch.nn.functional.linear(x, (1/self.lengthscale) * self.feature_weight, self.feature_bias))
        
    def forward(self, x):
        features = self.featurize(x)
        return features
    
    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.raw_lengthscale)
    
    @property
    def outputscale(self):
        return torch.nn.functional.softplus(self.raw_outputscale)
    
class RFFLaplace(RandomFourierFeatures):
    def __init__(self, in_features, out_features, rank=1024, lengthscale=20.0, outputscale=1.0, learnable_lengthscale=False, learnable_outputscale=False):
        super().__init__(in_features, rank, lengthscale, outputscale, learnable_lengthscale, learnable_outputscale)
        
        self.K = out_features
        self.linear = torch.nn.Linear(in_features=self.rank, out_features=self.K, bias=False)
        
        self.register_buffer("covariance", torch.zeros(self.K, self.rank, self.rank))
        
    def forward(self, x):
        features = self.featurize(x)
        logits = self.linear(features)
        return logits

    @torch.no_grad()
    def update_covariance_from_dataloader(self, dataloader, device="cpu"):
        
        precision = torch.eye(self.rank, device=device).unsqueeze(0).repeat(self.K, 1, 1)
        
        for x, _ in dataloader:
            
            batch_size = len(x)
            
            x = x.to(device)

            features = self.featurize(x) # (N, R)
            logits = self.linear(features) # (N, R)
            probs = torch.softmax(logits, dim=1) # (N, R)

            for i in range(batch_size):
                phi = features[i].unsqueeze(1) # (R, 1)
                # Update posterior precision
                weight = (probs[i] * (1 - probs[i])).reshape(self.K, 1, 1) # (K, 1, 1)
                precision += weight * (phi @ phi.T).unsqueeze(0) # (K, R, R)
                
        self.covariance = torch.inverse(precision)
        
    @torch.no_grad()
    def predict_proba(self, x, num_samples=10):
        
        batch_size = len(x)
        
        features = self.featurize(x) # (N, R)
        logits = self.linear(features) # (N, R)

        probs_list = []
        
        for i in range(batch_size):
            phi = features[i].unsqueeze(1) # (R, 1)
            # Compute posterior variance
            var = (phi.T @ self.covariance @ phi).squeeze() # (K,)
            # Compute predictive distribution
            pred_dist = torch.distributions.normal.Normal(loc=logits[i], scale=torch.sqrt(var))
            samples = pred_dist.sample(sample_shape=(num_samples,)) # (S, K,)
            probs = torch.nn.functional.softmax(samples, dim=1) # (S, K,)
            probs_list.append(torch.mean(probs, dim=0)) # (K,)
            
        return torch.stack(probs_list, dim=0)
    
class VariationalLinear(torch.nn.Module):
    def __init__(self, layer, raw_sigma_q=None, use_posterior=False):
        super().__init__()
        self.layer = layer
        self.raw_sigma_q = raw_sigma_q
        self.use_posterior = use_posterior
                
    def forward(self, x):
        if self.training or self.use_posterior:
            variational_params = self._variational_params()
            variational_weight, variational_bias = self._unflatten(variational_params)
            return torch.nn.functional.linear(
                x, 
                variational_weight, 
                variational_bias,
            )
        return self.layer(x)
    
    def _variational_params(self):
        params = self._flatten()
        eps = torch.randn_like(params)
        sigma = torch.nn.functional.softplus(self.raw_sigma_q)
        return params + sigma * eps
            
    def _flatten(self):
        return torch.cat([param.view(-1) for param in [self.layer.weight, self.layer.bias] if param is not None])
    
    def _unflatten(self, params):
        out = []
        for param in [self.layer.weight, self.layer.bias]:
            out.append(None if param is None else params[:param.numel()].view_as(param))
            if param is not None:
                params = params[param.numel():]
        return out
     
class VariationalConv2d(VariationalLinear):
    def __init__(self, layer, raw_sigma_q=None, use_posterior=False):
        super().__init__(layer, raw_sigma_q, use_posterior)
        
    def forward(self, x):
        if self.training or self.use_posterior:
            variational_params = self._variational_params()
            variational_weight, variational_bias = self._unflatten(variational_params)
            return torch.nn.functional.conv2d(
                x,
                variational_weight,
                variational_bias,
                self.layer.stride,
                self.layer.padding,
                self.layer.dilation,
                self.layer.groups
            )
        
        return self.layer(x)

class VariationalBatchNorm2d(VariationalLinear):
    def __init__(self, layer, raw_sigma_q=None, use_posterior=False):
        super().__init__(layer, raw_sigma_q, use_posterior)
        
    def forward(self, x):
        if self.training or self.use_posterior:

            if self.layer.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.layer.momentum

            if self.layer.training and self.layer.track_running_stats:
                if self.layer.num_batches_tracked is not None:
                    self.layer.num_batches_tracked.add_(1)
                    if self.layer.momentum is None:
                        exponential_average_factor = 1.0 / float(self.layer.num_batches_tracked)
                    else:
                        exponential_average_factor = self.layer.momentum

            if self.layer.training:
                bn_training = True
            else:
                bn_training = (self.layer.running_mean is None) and (self.layer.running_var is None)

            variational_params = self._variational_params()
            variational_weight, variational_bias = self._unflatten(variational_params)
            return torch.nn.functional.batch_norm(
                x, 
                self.layer.running_mean if not self.layer.training or self.layer.track_running_stats else None, 
                self.layer.running_var if not self.layer.training or self.layer.track_running_stats else None, 
                variational_weight,
                variational_bias,
                bn_training, 
                exponential_average_factor, 
                self.layer.eps, 
            )
        
        return self.layer(x)
