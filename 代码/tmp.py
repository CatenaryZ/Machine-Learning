import numpy as np
from scipy.stats import multivariate_normal

class BasicMCMC:
    """Basic MCMC method (Metropolis-Hastings)"""
    
    def __init__(self, target_dist, proposal_std=1.0):
        self.target_dist = target_dist
        self.proposal_std = proposal_std
        
    def sample(self, n_samples, initial_state, burn_in=1000):
        current_state = initial_state
        samples = []
        accepted = 0
        
        for i in range(n_samples + burn_in):
            # Generate candidate sample (normal distribution proposal)
            proposal = current_state + np.random.normal(0, self.proposal_std, size=current_state.shape)
            
            # Calculate acceptance probability
            current_prob = self.target_dist(current_state)
            proposal_prob = self.target_dist(proposal)
            acceptance_ratio = min(1, proposal_prob / current_prob)
            
            # Accept or reject
            if np.random.rand() < acceptance_ratio:
                current_state = proposal
                if i >= burn_in:
                    accepted += 1
            
            if i >= burn_in:
                samples.append(current_state.copy())
        
        acceptance_rate = accepted / n_samples
        return np.array(samples), acceptance_rate

class HamiltonianMC:
    """Hamiltonian Monte Carlo method"""
    
    def __init__(self, target_dist, step_size=0.1, n_leapfrog=10, mass=1.0):
        self.target_dist = target_dist
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.mass = mass
        
    def kinetic_energy(self, momentum):
        """Kinetic energy"""
        return 0.5 * np.sum(momentum**2) / self.mass
    
    def potential_energy(self, position):
        """Potential energy (negative log probability)"""
        return -np.log(self.target_dist(position) + 1e-10)
    
    def hamiltonian(self, position, momentum):
        """Hamiltonian"""
        return self.potential_energy(position) + self.kinetic_energy(momentum)
    
    def leapfrog_step(self, position, momentum):
        """Leapfrog integration step"""
        # Half step update momentum
        grad = self.numerical_gradient(position)
        momentum = momentum - 0.5 * self.step_size * grad
        
        # Full step update position
        position = position + self.step_size * momentum / self.mass
        
        # Half step update momentum
        grad = self.numerical_gradient(position)
        momentum = momentum - 0.5 * self.step_size * grad
        
        return position, momentum
    
    def numerical_gradient(self, position, eps=1e-6):
        """Numerical gradient calculation"""
        grad = np.zeros_like(position)
        for i in range(len(position)):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += eps
            pos_minus[i] -= eps
            grad[i] = (self.potential_energy(pos_plus) - self.potential_energy(pos_minus)) / (2 * eps)
        return grad
    
    def sample(self, n_samples, initial_state, burn_in=1000):
        current_state = initial_state
        samples = []
        accepted = 0
        
        for i in range(n_samples + burn_in):
            # Sample momentum from normal distribution
            current_momentum = np.random.normal(0, np.sqrt(self.mass), size=current_state.shape)
            
            # Leapfrog integration to simulate trajectory
            proposed_state = current_state.copy()
            proposed_momentum = current_momentum.copy()
            
            for _ in range(self.n_leapfrog):
                proposed_state, proposed_momentum = self.leapfrog_step(proposed_state, proposed_momentum)
            
            # Calculate Hamiltonian
            current_hamiltonian = self.hamiltonian(current_state, current_momentum)
            proposed_hamiltonian = self.hamiltonian(proposed_state, -proposed_momentum)  # Momentum reversal
            
            # Acceptance probability
            acceptance_ratio = min(1, np.exp(current_hamiltonian - proposed_hamiltonian))
            
            # Accept or reject
            if np.random.rand() < acceptance_ratio:
                current_state = proposed_state
                if i >= burn_in:
                    accepted += 1
            
            if i >= burn_in:
                samples.append(current_state.copy())
        
        acceptance_rate = accepted / n_samples
        return np.array(samples), acceptance_rate

class LangevinDynamics:
    """Langevin Dynamics MCMC"""
    
    def __init__(self, target_dist, step_size=0.1, mass=1.0):
        self.target_dist = target_dist
        self.step_size = step_size
        self.mass = mass
        
    def potential_energy(self, position):
        """Potential energy (negative log probability)"""
        return -np.log(self.target_dist(position) + 1e-10)
    
    def numerical_gradient(self, position, eps=1e-6):
        """Numerical gradient calculation"""
        grad = np.zeros_like(position)
        for i in range(len(position)):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += eps
            pos_minus[i] -= eps
            grad[i] = (self.potential_energy(pos_plus) - self.potential_energy(pos_minus)) / (2 * eps)
        return grad
    
    def sample(self, n_samples, initial_state, burn_in=1000):
        current_state = initial_state
        samples = []
        
        for i in range(n_samples + burn_in):
            # Langevin dynamics update
            grad = self.numerical_gradient(current_state)
            noise = np.random.normal(0, np.sqrt(2 * self.step_size), size=current_state.shape)
            
            current_state = current_state - self.step_size * grad + noise
            
            if i >= burn_in:
                samples.append(current_state.copy())
        
        # Langevin dynamics typically has high acceptance rate (close to 1)
        acceptance_rate = 1.0  # Simplified estimation
        return np.array(samples), acceptance_rate

# Testing and comparison
def main():
    # Set up target distribution: 2D Gaussian
    mean = np.array([1.0, -1.0])
    cov = np.array([[2.0, 0.8], [0.8, 1.5]])
    target_dist = multivariate_normal(mean, cov)
    
    def target_pdf(x):
        return target_dist.pdf(x)
    
    # Parameter settings
    n_samples = 5000
    burn_in = 1000
    initial_state = np.array([0.0, 0.0])
    
    # Basic MCMC
    print("Running Basic MCMC...")
    basic_mcmc = BasicMCMC(target_pdf, proposal_std=1.0)
    samples_basic, acc_basic = basic_mcmc.sample(n_samples, initial_state, burn_in)
    
    # Hamiltonian Monte Carlo
    print("Running Hamiltonian Monte Carlo...")
    hmc = HamiltonianMC(target_pdf, step_size=0.1, n_leapfrog=10)
    samples_hmc, acc_hmc = hmc.sample(n_samples, initial_state, burn_in)
    
    # Langevin Dynamics
    print("Running Langevin Dynamics...")
    langevin = LangevinDynamics(target_pdf, step_size=0.01)
    samples_langevin, acc_langevin = langevin.sample(n_samples, initial_state, burn_in)
    
    # Print acceptance rate comparison
    print("\n=== Acceptance Rate Comparison ===")
    print(f"Basic MCMC: {acc_basic:.3f}")
    print(f"HMC: {acc_hmc:.3f}")
    print(f"Langevin Dynamics: {acc_langevin:.3f}")

if __name__ == "__main__":
    main()