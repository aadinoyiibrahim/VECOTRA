"""
VECOTRA : VElocity Constrained Optinmal TRAnsport
    Constributors:
                Abdullahi Ibrahim
                Caterina De Bacco
"""

#######################################
# PACKAGES
#######################################

import pickle, time, warnings
import numpy as np
import networkx as nx
import random
import copy
import math
import scipy as sp
import itertools

from scipy.sparse import csr_matrix, lil_matrix, issparse, csc_matrix
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy import sparse

import quadprog
import scipy.optimize
import scipy.stats
import osqp
import scipy as scp

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#######################################
INF = 1e20

warnings.filterwarnings("ignore", message="Matrix is exactly singular")


def initialize_mu(flux, pflux, length, ord_norm=2):

    flux_norm = np.linalg.norm(flux, axis=1, ord=ord_norm)
    num = flux_norm ** (2 / (3 - pflux))
    return num


def tdensinit(nedge, seed=10, tdens_init=None):
    """initialization of the conductivities: mu_e ~ U(0,1)"""
    prng = np.random.RandomState(seed=seed)

    if tdens_init is None:
        tdens_0 = prng.uniform(0, 1, size=nedge)
    else:
        tdens_0 = 0.0001 * prng.uniform(0, 1, size=nedge) + tdens_init

    weight = np.ones(nedge) + 0.01 * prng.uniform(0, 1, size=nedge)

    return tdens_0, weight


def dyn(self,capacity):

    """dynamics method"""
    nodes, nedge = self.g.nodes(), self.g.number_of_edges()
    pflux = self.pflux
    forcing0 = self.forcing
    verbose = self.verbose
    g = self.g
    length = self.length
    tdens0=None
    coupling="l2"
    constraint = self.constraint_mode
    clipping = self.clipping_mode

    capacity = np.full(nedge, capacity)  # set capacity on every edges
    
    print("\ndynamics...")

    relax_linsys = 1.0e-5  # relaxation for stiffness matrix
    N_real = 5 # Number of iterations
    time_step = 0.09
    alpha = 2
    tot_time = 2000 # upper bound on number of time steps
    tol_var_tdens = 10e-3  
    threshold_cost = 1.0e-6  # threshold for stopping criteria using cost
    seed = self.seedG + self.seedF
    prng = np.random.RandomState(seed=seed )  # only needed if spsolve has problems (inside update)
    
    nnode = len(nodes)
    ncomm = forcing0.shape[0]
    forcing = forcing0.transpose()

    minCost = 1e14
    minCost_list = []

    inc_mat = csr_matrix(nx.incidence_matrix(g, nodelist=nodes, oriented=True))  # B
    inc_transpose = csr_matrix(inc_mat.transpose())  # B^T
    inv_len_mat = diags(1 / length, 0)  # diag[1/l_e]
    
    for r in range(N_real):

        tdens_0, weight = tdensinit( nedge, seed=seed + r, tdens_init=tdens0)  # conductivities initialization

        # --------------------------------------------------------------------------------
        tdens = tdens_0.copy()
        td_mat = diags(tdens.astype(float), 0)  # matrix M
        
        stiff =  inc_mat * td_mat * inv_len_mat * inc_transpose # B diag[mu] diag[1/l_e] B^T
        stiff_relax = stiff + relax_linsys * identity(nnode)  # avoid zero kernel
        pot = spsolve(stiff_relax, forcing).reshape((nnode, ncomm))  # pressure

        # --------------------------------------------------------------------------------
        # Run dynamics
        convergence_achieved = False
        cost_update = 0
        cost_list = []
        g_list = []
        time_iteration = 0

        fmax = forcing0.max()

        not_in_C_id = np.where(compute_capacity_contraint(tdens, capacity) < 0)[0]
        while not convergence_achieved and time_iteration < tot_time:

            time_iteration += 1

            # update tdens-pot system
            tdens_old = tdens.copy()
            pot_old = pot.copy()

            # print(f'mu:{tdens}')
            # equations update
            tdens, pot, grad, info = update(
                g,
                tdens,
                pot,
                weight,
                inc_mat,
                inc_transpose,
                inv_len_mat,
                forcing,
                time_step,
                pflux,
                relax_linsys,
                nnode,
                coupling=coupling,
                constraint=constraint,
                clipping=clipping,
                capacity=capacity,
                alpha=alpha,
            )

            g_ = compute_capacity_contraint(tdens, capacity)
            g_list.append(g_)

            # print(f'g_list:{g_list}')
            # singular stiffness matrix
            if info != 0:
                print(f"info = {info}")
                tdens = (
                    tdens_old + prng.rand(*tdens.shape) * np.mean(tdens_old) / 1000.0
                )
                pot = pot_old + prng.rand(*pot.shape) * np.mean(pot_old) / 1000.0

            # 1) convergence with conductivities
            # var_tdens = max(np.abs(tdens - tdens_old))/time_step
            # print(time_iteration, var_tdens)
            if verbose > 1:
                print("==========")

            # 2) an alternative convergence criteria: using total cost and maximum variation of conductivities
            var_tdens = max(np.abs(tdens - tdens_old)) / time_step

            # var_tdens_inter = ([ max(np.abs(tdens_inter[i] - tdens_old_inter[i]))/time_step for i in range(nnode) ] )
            (
                convergence_achieved,
                cost_update,
                abs_diff_cost,
                flux_norm,
                flux_mat,
            ) = cost_convergence(
                threshold_cost,
                cost_update,
                tdens,
                pot,
                inc_mat,
                inv_len_mat,
                length,
                weight,
                pflux,
                convergence_achieved,
                var_tdens,
                coupling=coupling,
            )  # , var_tdens_inter)

            if verbose > 1:
                # print(time_iteration, var_tdens/forcing.max(), abs_diff_cost)

                # print('\r','It=',it,'err=', abs_diff,'J-J_old=',abs_diff_cost,sep=' ', end='', flush=True)
                print(
                    "\r",
                    "it=%3d, err/max_f=%5.2f, J_diff=%8.2e"
                    % (time_iteration, var_tdens / fmax, abs_diff_cost),
                    sep=" ",
                    end=" ",
                    flush=True,
                )
                time.sleep(0.05)

            cost_list.append(cost_update)

            if var_tdens < tol_var_tdens:  # or (var_tdens_inter < tol_var_tdens):
                convergence_achieved = True

            elif time_iteration >= tot_time:
                convergence_achieved = True
                tdens = tdens_old.copy()
                if verbose > 0:
                    print(
                        "ERROR: convergence dynamics not achieved, iteration time > maxit"
                    )

            if convergence_achieved == True and time_iteration < tot_time:
                if constraint == True:  # check that all constraints are satisfied
                    not_in_C_id = np.where(
                        compute_capacity_contraint(tdens, capacity) <= 0
                    )[0]
                    if len(not_in_C_id) > 0:
                        convergence_achieved = False


        if convergence_achieved:
            if verbose > 0:
                print("cost:", cost_update, " - N_real:", r, "- Best cost", minCost)
            if cost_update < minCost:
                minCost = cost_update
                minCost_list = cost_list
                tdens_best = tdens.copy()
                pot_best = pot.copy()
                flux_best_norm = flux_norm.copy()
                flux_best = flux_mat.copy()
                min_g_list = g_list.copy()

        else:
            print("ERROR: convergence dynamics not achieved")

    return tdens_best,pot_best,flux_best_norm,minCost,minCost_list,flux_best, min_g_list

def update(
    g,
    tdens: np.ndarray,
    pot: np.ndarray,
    weight: np.ndarray,
    inc_mat: csr_matrix,
    inc_transpose: csr_matrix,
    inv_len_mat: diags,
    forcing: np.ndarray,
    time_step: float,
    pflux: float,
    relax_linsys: float,
    nnode: int,
    coupling="l1",
    constraint: bool =True,
    clipping: bool =False,
    capacity: np.ndarray =None,
    alpha: int =1,
):
    """
    ---------------
    dynamics update
    ---------------
        Returns:
            self.tdens: np.array, updated conductivities
            grad : np.array, discrete gradient
            pot: np.array, updated potential matrix on nodes
            info: bool, sanity check 
    """

    

    nedge = tdens.shape[0]
    nnode, ncomm = pot.shape

    weight = 1.0
    grad = inv_len_mat * inc_transpose * pot  # discrete gradient
    
    if coupling == "l2":
        rhs_ode = rhs_ode = (
            (tdens ** pflux) * ((grad ** 2).sum(axis=1)) / (weight ** 2)
        ) - tdens
    if coupling == "l1":
        rhs_ode = rhs_ode = (
            (tdens ** pflux) * ((np.abs(grad).sum(axis=1)) ** 2) / (weight ** 2)
        ) - tdens
   
    if clipping == True:  # use trivial method of clipping
        constraint = False

    # --------------------------
    # adding force to enforce constraints
    # --------------------------

    if constraint == True:
        """
        this section runs analytic result
        """
        small_err = 1e-05
        g_capacity = compute_capacity_contraint(tdens, capacity)
        not_in_C_id = np.where(g_capacity <= 0)[0]
        alpha_pos = 1 / time_step

        if len(not_in_C_id) > 0:
            for i in np.arange(nedge):
                if rhs_ode[i] >= alpha * g_capacity[i]:
                    rhs_ode[i] = alpha * g_capacity[i]
                else:
                    rhs_ode[i] = rhs_ode[i] 
        
    # update conductivity
    if rhs_ode.ndim > 1:
        if alpha == 0:
            tdens = tdens + time_step * np.ravel(rhs_ode[:, 0])
        else:
            tdens = tdens + (alpha * time_step * np.ravel(rhs_ode[:, 0]))
    else:
        if alpha == 0:
            tdens = tdens + time_step * rhs_ode
        else:
            tdens = tdens + (alpha * time_step * rhs_ode)

    n_negative = np.sum(tdens < 0)
    if n_negative > 0:
        print(f"min tdens = {min(tdens)}--clip to:{.1 * np.min(tdens[tdens>0])}")
        tdens[tdens < 0] = 0.1 * np.min(tdens[tdens > 0])

    if clipping == True:  # clip tdens to capacities
        not_in_C_id = np.where(compute_capacity_contraint(tdens, capacity) < 0)[0]
        if len(not_in_C_id) > 0:
            tdens[not_in_C_id] = capacity[not_in_C_id]
            not_in_C_id = np.where(compute_capacity_contraint(tdens, capacity) < 0)[0]

    td_mat = diags(tdens.astype(float), 0)

    # update stiffness matrix
    stiff = inc_mat * td_mat * inv_len_mat * inc_transpose

    # spsolve
    stiff_relax = stiff + relax_linsys * identity(nnode)  # avoid zero kernel
    # update potential
    pot = spsolve(stiff_relax, forcing).reshape((nnode, ncomm))  # pressure
    if np.any(np.isnan(pot)):  # or np.any(np.isnan(pot_ctr)): # or np.any(pot != pot)
        info = -1
        pass
    else:
        info = 0

    return tdens, pot, grad, info


def cost_convergence(
    threshold_cost: float,
    cost: int,
    tdens: np.ndarray,
    pot: np.ndarray,
    inc_mat: csr_matrix,
    inv_len_mat: diags,
    length: np.ndarray,
    weight: np.ndarray,
    pflux: float,
    convergence_achieved: bool,
    var_tdens: np.float64,
    coupling: str ="l1",
):  
    """
    ---------------------
    computing convergence
    --------------------- 
    using total cost, setting a high value for maximum conductivity variability
    
    Returns:
            
            convergence_achieved: np.array, updated convergence condition
            cost_update: np.float65, updated cost
            abs_diff_cost: np.float65, cost differences
            flux_norm: np.array, normalized fluxes
            flux_mat: np.array, fluxes matrix
    """
    

    L = len(pot)
    nnode = pot[0].shape[0]

    td_mat = np.diag(tdens.astype(float))

    flux_mat = np.matmul(td_mat * inv_len_mat * np.transpose(inc_mat), pot)

    if coupling == "l2":
        flux_norm = np.linalg.norm(flux_mat, axis=1)
    if coupling == "l1":
        flux_norm = np.linalg.norm(flux_mat, axis=1, ord=1)

    cost_update = np.sum(length * (flux_norm ** (2 * (2 - pflux) / (3 - pflux))))

    abs_diff_cost = abs(cost_update - cost)

    convergence_achieved = bool(convergence_achieved)
    if pflux > 0.0: # sanity check
        if abs_diff_cost < threshold_cost:
            convergence_achieved = True
    else:
        if abs_diff_cost < threshold_cost and var_tdens < 1:
            convergence_achieved = True
    
    return convergence_achieved, cost_update, abs_diff_cost, flux_norm, flux_mat

def compute_capacity_contraint(tdens: np.ndarray, capacities: np.ndarray):

    """
    tdens: np.ndarray,
    capacities: np.ndarray,

    Returns:
            g = c - mu
    """
    return capacities - tdens


