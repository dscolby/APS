import torch
from gpytorch import constraints
from gpytorch.kernels import Kernel

class PolicyKernel(Kernel):
    def __init__(self, state_size: int, action_size: int, num_samples: int=10):
        super(PolicyKernel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.num_samples = num_samples

        self.register_parameter(name="raw_lengthscale", 
                                parameter=torch.nn.Parameter(torch.tensor(1e-6, 
                                                                          requires_grad=True)))
        self.register_parameter(name="raw_variance", 
                                parameter=torch.nn.Parameter(torch.tensor(1e-6, 
                                                                          requires_grad=True)))
        self.raw_lengthscale_constraint = constraints.Positive()
        self.raw_variance_constraint = constraints.Positive()

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)

    @variance.setter
    def variance(self, value):
        self._set_variance(value)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)  # Convert to batch of size 1
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
            
        random_states = (-1.0 - 1.0) * torch.rand(self.state_size, 
                                                  self.num_samples).to(torch.float64) + 1.0
        random_states = random_states.unsqueeze(-1).permute(1, 0, 2)

        # Making x1 and x2 into num_samples x action_dims x state_dims
        x1 = x1.reshape((-1, self.action_size, self.state_size))
        x2 = x2.reshape((-1, self.action_size, self.state_size))

        actions1 = torch.einsum('abc,jkl->ajb', x1, random_states)
        actions2 = torch.einsum('abc,jkl->ajb', x2, random_states)
        
        # Calculate pairwise mean distances between policies
        actions1, actions2 = actions1.unsqueeze(1), actions2.unsqueeze(0)
        dists = torch.pow(actions1 - actions2, 2)
        dists = dists.sum(dim=-1)
        dists = dists.mean(dim=-1)

        kernel_matrix = self.variance * torch.exp(-dists / self.lengthscale)

        if diag:
            kernel_matrix = kernel_matrix[0]

        return kernel_matrix