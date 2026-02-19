import numpy as np
from bml.samplers.utils import leapfrog, find_reasonable_epsilon


class DualAveragingHMC():
    def __init__(self, L, grad):
        self.L = L
        self.grad = grad
    
    def sample(self, theta0, delta, lam, M, M_adapt):
        """Runs the Dual Averaging HMC sampler."""
        # Dual averaging parameters
        epsilon = find_reasonable_epsilon(theta0, self.grad, self.L)
        mu = np.log(10 * epsilon)
        epsilon_bar  = 1.0
        H_bar = 0.0
        gamma = 0.05
        t0 = 10
        kappa = 0.75

        theta_prev = theta0
        samples = [theta0]

        for m in range(1, M+1):
            # Sample momentum
            r0 = np.random.normal(size=theta_prev.shape)

            theta_next = theta_prev
            theta_tilde = theta_prev
            r_tilde = r0
            L_m = max(1, round(lam / epsilon))

            for _ in range(1, L_m+1):
                theta_tilde, r_tilde = leapfrog(theta_tilde, r_tilde, epsilon, self.grad)

            alpha = min(1, np.exp(self.L(theta_tilde) - 0.5 * np.dot(r_tilde, r_tilde)) / np.exp(self.L(theta_prev) - 0.5 * np.dot(r0, r0)))
            if np.random.rand() < alpha:
                theta_next = theta_tilde
                r_next = -r_tilde
            
            if m <= M_adapt:
                H_bar = (1 - 1/(m + t0)) * H_bar + (1/(m + t0)) * (delta - alpha)
                log_epsilon = mu - (np.sqrt(m) / gamma) * H_bar
                epsilon = np.exp(log_epsilon)
                log_epsilon_bar = m ** (-kappa) * log_epsilon + (1 - m ** (-kappa)) * np.log(epsilon_bar)
                epsilon_bar = np.exp(log_epsilon_bar)
            
            else:
                epsilon = epsilon_bar

            samples.append(theta_next)
            theta_prev = theta_next
        
        return np.array(samples)

