import torch
from torch import Tensor


from botorch import settings
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective, IdentityMCObjective
from botorch.models.gpytorch import GPyTorchModel

from botorch.models.model import Model
from botorch.models import SingleTaskGP
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf_cyclic, optimize_acqf
from botorch.optim.initializers import initialize_q_batch_nonneg

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, Kernel, ProductKernel, AdditiveKernel, GridInterpolationKernel, AdditiveStructureKernel, ProductStructureKernel
from gpytorch.utils.grid import choose_grid_size

from typing import Optional, Union



class qEISP(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        mc_points: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[ScalarizedObjective] = None,
        X_pending: Optional[Tensor] = None,
        maximize: bool = True,
    ) -> None:
        r"""q-Espected Improvement of Skill Performance. 

        Args:
            model: A fitted model.
            beta: value to trade off between upper confidence bound and mean of fantasized performance.
            mc_points: A `batch_shape x N x d` tensor of points to use for
                MC-integrating the posterior variance. Usually, these are qMC
                samples on the whole design space, but biased sampling directly
                allows weighted integration of the posterior variance.
            sampler: The sampler used for drawing fantasy samples. In the basic setting
                of a standard GP (default) this is a dummy, since the variance of the
                model after conditioning does not actually depend on the sampled values.
            objective: A ScalarizedObjective. Required for multi-output models.
            X_pending: A `n' x d`-dim Tensor of `n'` design points that have
                points that have been submitted for function evaluation but
                have not yet been evaluated.
            maximize: If true uses the UCB of performance scaled by beta, else it uses LCB

            Docstring from BOTorch class and same with comments below
        """
        super().__init__(model=model, objective=objective)
        if sampler is None:
            # If no sampler is provided, we use the following dummy sampler for the
            # fantasize() method in forward. IMPORTANT: This assumes that the posterior
            # variance does not depend on the samples y (only on x), which is true for
            # standard GP models, but not in general (e.g. for other likelihoods or
            # heteroskedastic GPs using a separate noise model fit on data).
            sampler = SobolQMCNormalSampler(
                num_samples=1, resample=False, collapse_batch_dims=True
            )
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)
        self.sampler = sampler
        self.X_pending = X_pending
        self.register_buffer("mc_points", mc_points)
        self.maximize = maximize

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self.beta = self.beta.to(X)
        with settings.propagate_grads(True):
            posterior = self.model.posterior(X=X)
            batch_shape = X.shape[:-2]
            mean = posterior.mean.view(*batch_shape, X.shape[-2], -1)
            variance = posterior.variance.view(*batch_shape, X.shape[-2], -1)
            delta = self.beta.expand_as(mean) * variance.sqrt()
        
            if self.maximize:
                Yhat = mean + delta
            else:
                Yhat = mean - delta
            
            bdims = tuple(1 for _ in X.shape[:-2])
            if self.model.num_outputs > 1:
                # We use q=1 here b/c ScalarizedObjective currently does not fully exploit
                # lazy tensor operations and thus may be slow / overly memory-hungry.
                # TODO (T52818288): Properly use lazy tensors in scalarize_posterior
                mc_points = self.mc_points.view(-1, *bdims, 1, X.size(-1))
            else:
                # While we only need marginal variances, we can evaluate for q>1
                # b/c for GPyTorch models lazy evaluation can make this quite a bit
                # faster than evaluting in t-batch mode with q-batch size of 1
                mc_points = self.mc_points.view(*bdims, -1, X.size(-1))
                
            Yhat = Yhat.view(*batch_shape, X.shape[-2], -1)
            
            fantasy_model = self.model.condition_on_observations(X=X, Y=Yhat)
                
            posterior1 = self.model.posterior(mc_points)
            posterior2 = fantasy_model.posterior(mc_points)
            
            # transform with the scalarized objective
            posterior1 = self.objective(posterior1.mean)
            posterior2 = self.objective(posterior2.mean)

            improvement = posterior2 - posterior1

            return improvement.mean(dim=-1)



class StandardActiveLearningGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API
    
    def __init__(self, train_X, train_Y, bounds=None):
        # squeeze output dim before passing train_Y to ExactGP
        super(StandardActiveLearningGP, self).__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        xdims = train_X.shape[-1]
        self.Kspatial = ScaleKernel(RBFKernel(active_dims=torch.tensor(list(range(xdims-1)))))
        self.Ktime = ScaleKernel(RBFKernel(active_dims=torch.tensor([xdims-1])))
        # Kspatial = ScaleKernel(RBFKernel())
        # Ktime = ScaleKernel(RBFKernel())
        
        # self.covar_module = ScaleKernel(RBFKernel()) # AdditiveKernel(Kspatial, ProductKernel(Kspatial, Ktime))
        self.covar_module = AdditiveKernel(self.Kspatial, ProductKernel(self.Kspatial, self.Ktime))
        self.to(train_X)  # make sure we're on the right device/dtype
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class ActiveLearningTaskSampler(object):
    def __init__(self,ranges):
        self.ranges = ranges
        self.xdim = ranges.shape[0] + 1
        self.model = None
        self.mll = None
        self.Xdata = None
        self.Ydata = None
        
        self.bounds = torch.tensor(ranges)
        self.bounds = torch.cat([self.bounds, torch.tensor([[0.0,1.0]])]).T
        
        

    def update_model(self, new_X, new_Y, refit=False):
        if self.model is not None:
            new_X = new_X.to(self.X)
            new_Y = new_Y.to(self.X)
            self.X = torch.cat([self.X, new_X.to(self.X)])
            
            self.Y = torch.cat([self.Y, new_Y.to(self.X)])
            state_dict = self.model.state_dict()
        else:
            self.X = new_X.float()
            self.Y = new_Y.float()
            state_dict = None
        
        T = 12*50
        if self.X.shape[0] >= T:
            self.X = self.X[-T:, :]
            self.Y = self.Y[-T:, :]

        if self.X.shape[0] < 5:  # TODO seems to throw an error if only one sample is present. Refitting should probably only happen every N data points anyways
            return None

        if refit:
            model = StandardActiveLearningGP(self.X, self.Y, bounds=self.bounds)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            self.model = model
            self.mll = mll
            if state_dict is not None:
                self.model.load_state_dict(state_dict)
            fit_gpytorch_model(mll)
        else:
            self.model.set_train_data(self.X, self.Y)
            # self.model = self.model.condition_on_observations(new_X, new_Y)  # TODO: might be faster than setting the data need to test

    def get_design_points(self, num_points:int=1, time=None):
        if not self.model or time < 30:
            return sample_random_points(self.bounds, num_points)
        
        if not time:
            time = self.X[:, -1].max() + 1

        bounds = self.bounds
        bounds[:, -1] = time
        num_mc = 500
        mc_points = torch.rand(num_mc, bounds.size(1), device=self.X.device, dtype=self.X.dtype)
        mc_points = bounds[0] + (bounds[1] - bounds[0]) * mc_points
        
        qeisp = qEISP(self.model, mc_points=mc_points, beta=1.96)
        try:
            candidates, acq_value = optimize_acqf(
                acq_function=qeisp,
                bounds=bounds,
                raw_samples=128,
                q=num_points,
                num_restarts=1,
                return_best_only=True,
            )
            return candidates
        except: 
            return sample_random_points(self.bounds, num_points)


def sample_random_points(bounds, num_points):
    points = torch.rand(num_points, bounds.size(1), device=bounds.device, dtype=bounds.dtype)
    points = bounds[0] + (bounds[1] - bounds[0]) * points
    return points
    