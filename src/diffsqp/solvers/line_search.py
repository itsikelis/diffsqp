"""Batched acceptance tests for the SQP backtracking line search.

Every strategy sees, per batch element, the cost f and the violations of the
current iterate and of each trial point, and decides which trial points to
accept. The constraint violation is theta = max(dyn_inf, constr_inf).

    "filter"         legacy: accept if cost, dynamics violation or constraint
                     violation improves on the current iterate.
    "merit"          legacy: phi = f + merit_mu * theta with a fixed merit_mu,
                     Armijo test on the cost gradient only.
    "filter_ipopt"   filter line search of Waechter & Biegler (2006), with a
                     filter per batch element.
    "merit_adaptive" exact penalty phi = f + mu * theta, with the directional
                     derivative grad f . d - mu * theta and a per element mu
                     from Nocedal & Wright (2006), eq. 18.36.
"""

import torch


def violation(dyn_inf, constr_inf):
    return torch.maximum(dyn_inf, constr_inf)


class LineSearch:
    def begin(self, sqp_iter, cost, dyn_inf, constr_inf, grad_dot_d, dHd, theta_lin):
        """Called once per SQP iteration, before the trial points.

        grad_dot_d: directional derivative of the cost along the QP step.
        dHd: curvature of the QP model along the step (d^T H d).
        theta_lin: violation of the linearized constraints after the full step
            (zero for an exact QP solution).
        """

    def accept(self, alpha, cost, dyn_inf, constr_inf, active):
        """Mask of accepted trial points among the `active` elements."""
        raise NotImplementedError

    def end(self):
        """Called once per SQP iteration, after the line search."""


class LegacyFilter(LineSearch):
    def begin(self, sqp_iter, cost, dyn_inf, constr_inf, grad_dot_d, dHd, theta_lin):
        self.current = (cost, dyn_inf, constr_inf)

    def accept(self, alpha, cost, dyn_inf, constr_inf, active):
        cost_0, dyn_0, constr_0 = self.current
        improved = (cost < cost_0) | (dyn_inf < dyn_0) | (constr_inf < constr_0)
        return improved & active


class LegacyMerit(LineSearch):
    def __init__(self, parameters):
        self.mu = parameters.merit_mu
        self.beta = parameters.armijo_beta

    def begin(self, sqp_iter, cost, dyn_inf, constr_inf, grad_dot_d, dHd, theta_lin):
        self.phi = cost + self.mu * violation(dyn_inf, constr_inf)
        self.dir_deriv = grad_dot_d

    def accept(self, alpha, cost, dyn_inf, constr_inf, active):
        phi = cost + self.mu * violation(dyn_inf, constr_inf)
        return (phi < self.phi + self.beta * alpha * self.dir_deriv) & active


class AdaptiveMerit(LineSearch):
    """phi = f + mu * theta with mu chosen so that the QP step is a descent
    direction of phi.

    The directional derivative of theta along the step is predicted by the
    linearization, theta_lin - theta (= -theta for an exact QP solution), so
    D(phi; d) = grad f . d - mu * (theta - theta_lin). Requiring
    mu >= (grad f . d + 0.5 d^T H d) / ((1 - rho) * (theta - theta_lin)) gives
    D(phi; d) <= -rho * mu * (theta - theta_lin) - 0.5 * d^T H d < 0.
    mu never decreases. If the step does not reduce the linearized violation,
    phi must not increase.
    """

    def __init__(self, parameters, batch_size):
        self.mu = torch.full((batch_size,), parameters.merit_mu_init)
        self.beta = parameters.armijo_beta
        self.rho = 0.1
        self.margin = 1.1

    def begin(self, sqp_iter, cost, dyn_inf, constr_inf, grad_dot_d, dHd, theta_lin):
        theta = violation(dyn_inf, constr_inf)
        predicted = theta - theta_lin
        reduces = predicted > 1e-12
        mu_required = (grad_dot_d + 0.5 * torch.clamp(dHd, min=0.0)) / (
            (1.0 - self.rho) * torch.clamp(predicted, min=1e-12)
        )
        raise_mu = reduces & (self.mu < mu_required)
        self.mu = torch.where(raise_mu, self.margin * mu_required, self.mu)

        self.phi = cost + self.mu * theta
        self.dir_deriv = torch.clamp(grad_dot_d - self.mu * predicted, max=0.0)

    def accept(self, alpha, cost, dyn_inf, constr_inf, active):
        phi = cost + self.mu * violation(dyn_inf, constr_inf)
        armijo = phi <= self.phi + self.beta * alpha * self.dir_deriv
        return armijo & torch.isfinite(phi) & active


class IpoptFilter(LineSearch):
    """Filter line search (Waechter & Biegler 2006, Algorithm A), batched.

    A trial point must not be dominated by any filter entry (with margins) and
    must sufficiently decrease theta or f with respect to the current iterate.
    When the iterate is nearly feasible and the step is a good descent
    direction of f (switching condition), an Armijo decrease of f is required
    instead and the filter is not augmented. Otherwise the current iterate is
    added to the filter once a step is accepted, which rules out cycling.
    """

    gamma_theta = 1e-5
    gamma_f = 1e-5
    delta = 1.0
    s_theta = 1.1
    s_f = 2.3

    def __init__(self, parameters, cost, dyn_inf, constr_inf):
        batch_size = cost.shape[0]
        theta_0 = torch.clamp(violation(dyn_inf, constr_inf), min=1.0)
        self.theta_max = 1e4 * theta_0
        self.theta_min = 1e-4 * theta_0
        self.beta = parameters.armijo_beta
        # Filter entries (theta, f) per element, one slot per SQP iteration.
        # Empty slots (theta = +inf) accept everything.
        n_slots = max(parameters.sqp_max_iter, 1)
        self.filter_theta = torch.full((batch_size, n_slots), float("inf"))
        self.filter_f = torch.full((batch_size, n_slots), float("inf"))

    def begin(self, sqp_iter, cost, dyn_inf, constr_inf, grad_dot_d, dHd, theta_lin):
        self.sqp_iter = sqp_iter
        self.f = cost
        self.theta = violation(dyn_inf, constr_inf)
        self.grad_dot_d = grad_dot_d
        self.accepted = torch.zeros_like(cost, dtype=torch.bool)
        self.f_type = torch.zeros_like(cost, dtype=torch.bool)

    def accept(self, alpha, cost, dyn_inf, constr_inf, active):
        theta = violation(dyn_inf, constr_inf)

        not_dominated = (
            (theta.unsqueeze(1) < self.filter_theta)
            | (cost.unsqueeze(1) < self.filter_f)
        ).all(dim=1)

        switching = (
            (self.grad_dot_d < 0)
            & (
                alpha * torch.pow(torch.clamp(-self.grad_dot_d, min=0.0), self.s_f)
                > self.delta * torch.pow(self.theta, self.s_theta)
            )
            & (self.theta <= self.theta_min)
        )
        armijo = cost <= self.f + self.beta * alpha * self.grad_dot_d
        sufficient = (theta <= (1.0 - self.gamma_theta) * self.theta) | (
            cost <= self.f - self.gamma_f * self.theta
        )

        accepted = (
            active
            & torch.isfinite(cost)
            & torch.isfinite(theta)
            & (theta <= self.theta_max)
            & not_dominated
            & torch.where(switching, armijo, sufficient)
        )
        self.f_type = torch.where(accepted, switching, self.f_type)
        self.accepted = self.accepted | accepted
        return accepted

    def end(self):
        # Steps that were not f-type add the (margined) current iterate
        augment = (self.accepted & ~self.f_type).unsqueeze(1)
        slot = self.sqp_iter % self.filter_theta.shape[1]
        theta_entry = ((1.0 - self.gamma_theta) * self.theta).unsqueeze(1)
        f_entry = (self.f - self.gamma_f * self.theta).unsqueeze(1)
        self.filter_theta[:, slot : slot + 1] = torch.where(
            augment, theta_entry, self.filter_theta[:, slot : slot + 1]
        )
        self.filter_f[:, slot : slot + 1] = torch.where(
            augment, f_entry, self.filter_f[:, slot : slot + 1]
        )


def make_line_search(parameters, batch_size, cost, dyn_inf, constr_inf):
    name = parameters.ls_function
    if name == "filter":
        return LegacyFilter()
    if name == "merit":
        return LegacyMerit(parameters)
    if name == "merit_adaptive":
        return AdaptiveMerit(parameters, batch_size)
    if name == "filter_ipopt":
        return IpoptFilter(parameters, cost, dyn_inf, constr_inf)
    raise ValueError(f"Unknown line search function: {name}")
