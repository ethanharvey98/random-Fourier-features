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
        