# PyTorch
import torch
# Importing our custom module(s)
import utils

class RandomFourierFeaturesGaussianProcess(torch.nn.Module):
    def __init__(self, in_features, out_features, learnable_lengthscale=False, learnable_outputscale=False, lengthscale=20.0, outputscale=1.0, rank=1024):
        super().__init__()
        
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("feature_weight", torch.randn(self.rank, in_features))
        self.register_buffer("feature_bias", 2 * torch.pi * torch.rand(self.rank))
        self.linear = torch.nn.Linear(in_features=self.rank, out_features=out_features, bias=False)
                
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
        logits = self.linear(features)
        return logits

    def predict_proba(self, x, num_samples=100):
        
        K = self.out_features
        batch_size = len(x)
        # Compute features
        Phi_NR = self.featurize(x)
        # Compute posterior mean
        logits_NK = self.linear(Phi_NR)

        probs_list = []
        for i in range(batch_size):
            phi_R1 = Phi_NR[i].unsqueeze(1)
            # Compute posterior variance
            var_K = (phi_R1.T @ self.covariance_KRR @ phi_R1).squeeze()
            # Compute predictive distribution
            distribution = torch.distributions.normal.Normal(loc=logits_NK[i], scale=torch.sqrt(var_K))
            samples_SK = distribution.sample(sample_shape=(num_samples,))
            probs_SK = torch.nn.functional.softmax(samples_SK, dim=1)
            probs_list.append(torch.mean(probs_SK, dim=0))
            
        return torch.stack(probs_list, dim=0)
    
    def update_covariance_from_dataloader(self, dataloader, device="cpu"):

        K = self.out_features
        precision_KRR = torch.eye(self.rank, device=device).unsqueeze(0).repeat(K, 1, 1)
        
        with torch.no_grad():

            for X, _ in dataloader:
                
                if device == "cuda":
                    X = X.to(device)
                
                batch_size = len(X)

                Phi_NR = self.featurize(X)
                logits_NK = self.linear(Phi_NR)
                probs_NK = torch.nn.functional.softmax(logits_NK, dim=1)
                
                for i in range(batch_size):
                    phi_R1 = Phi_NR[i].unsqueeze(1)
                    # Update posterior covariance
                    precision_KRR += (probs_NK[i] * (1 - probs_NK[i])).reshape(K, 1, 1) * (phi_R1 @ phi_R1.T).unsqueeze(0)
        # Compute posterior covariance
        self.covariance_KRR = torch.inverse(precision_KRR)
            
    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.raw_lengthscale)
    
    @property
    def outputscale(self):
        return torch.nn.functional.softplus(self.raw_outputscale)
