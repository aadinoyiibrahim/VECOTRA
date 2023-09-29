"""
VECOTRA : VElocity Constrained Optinmal TRAnsport
    Constributors:
                Abdullahi Ibrahim
                Caterina De Bacco
"""
from scipy.spatial import distance
import networkx as nx
import numpy as np
import pandas as pd
from init import *
from dyn import *
import generate_planar as gpl

import warnings

warnings.filterwarnings('ignore')

class VECOTRA:
    '''
    Constrained Optimal Transport
    '''

    def __init__(self, seedG, seedF, pflux, 
                 N, p, weigth, out_folder, alpha, verbose,
                 constraint_mode, clipping_mode, plot_cost):

        self.seedG = seedG
        self.N = N
        self.p = p
        self.pflux = pflux
        self.seedF = seedF
        self.weigth = weigth
        self.verbose = verbose
        self.g = gpl.planar_graph(self.N, L_min=0.0, L_max=None, domain=(0, 0, 1, 1), seed=self.seedG)
        self.out_folder = out_folder
        self.alpha = alpha
        self.constraint_mode = constraint_mode
        self.clipping_mode = clipping_mode
        self.forcing = None
        self.length = np.zeros(self.g.number_of_edges())
        nedge = self.g.number_of_edges()
        init_capa = 1e20
        self.capacity = np.full(nedge, init_capa)
        
        # results
        self.tdens_best = []
        self.pot_best = []
        self.flux_best_norm = []
        self.minCost = []
        self.minCost_list = []
        self.flux_best = []
        self.min_g_list = []

    def assign_length(self):
        '''
        Assign lengths to edges in the graph.
        '''
        label = f'synt_{self.N}_{self.p}_{self.pflux}'
        coord_file_name = f'{label}{self.seedG}coord.csv'
        nodes = list(self.g.nodes())
        input_path = self.out_folder
        file_edges = f'{label}{self.seedG}.csv'
        df = from_nx_graphs2df([self.g], outfile=input_path + file_edges)
        edges_file_name = input_path + file_edges
        graph, nnode, nedge, nodes, nodes_inter, nodeName2Id, nodeId2Name = file2graph(edges_file_name, sep=',')
        df_coord = from_nx_graph2coord([self.g], nodes, outfile=input_path + coord_file_name,
                                       mapping=nodeName2Id, nodes_inter=nodes_inter)
        coord_importing(input_path + coord_file_name, self.g, nid='nodeID', nlat='nodeLat', nlon='nodeLong', sep=',')
        self.length = compute_euclidean(self.g)
        length_dict = {e: self.length[idx] for idx, e in enumerate(list(self.g.edges()))}
        nx.set_edge_attributes(self.g, length_dict, 'length')
        return self.length, self.g

    def generate_forcing(self):
        '''
        Generate forcing (O-D pairs) for the graph.
        '''
        G = gpl.planar_graph(self.N, L_min=0.0, L_max=None, domain=(0, 0, 1, 1), seed=self.seedG)
        forcing = forcing_generate(self.g, self.p, self.weigth, pos=None, pos_center_of_mass=None, seed=self.seedF)
        rhs_df = pd.DataFrame(forcing, columns=['source', 'sink', 'weight'])
        label = f'synt_{self.N}_{self.p}_{self.pflux}'
        out_forcing = f'{self.out_folder}{label}_forcing{self.seedF}.csv'
        rhs_df.to_csv(out_forcing, sep=' ', header=True, index=False)
        print(f'destination:{np.unique(rhs_df["sink"])} -- weight:{self.weigth}')
        input_path = self.out_folder
        file_edges = f'{label}{self.seedG}.csv'
        df = from_nx_graphs2df([G], outfile=input_path + file_edges)
        edges_file_name = input_path + file_edges
        graph, nnode, nedge, nodes, nodes_inter, nodeName2Id, nodeId2Name = file2graph(edges_file_name, sep=',')
        input_forcing = out_forcing
        self.forcing, comm_list, transit_list = forcing_importing_from_file(input_forcing, nodes, nodes_inter,
                                                                           nodeName2Id, sep=' ', header=0, source='source',
                                                                           sink='sink')
        self.g = graph.copy()
        coord_file_name = f'{label}{self.seedG}coord.csv'
        df_coord = from_nx_graph2coord([G], nodes, outfile=input_path + coord_file_name,
                                       mapping=nodeName2Id, nodes_inter=nodes_inter)
        coord_importing(input_path + coord_file_name, self.g, nid='nodeID', nlat='nodeLat', nlon='nodeLong', sep=',')
        self.length = compute_euclidean(self.g)
        length_dict = {e: self.length[idx] for idx, e in enumerate(list(self.g.edges()))}
        nx.set_edge_attributes(self.g, length_dict, 'length')
        return self.g, self.length, self.forcing, comm_list, transit_list

    def exec_uconstr(self, capacity, constraint_mode, clipping_mode):
        '''
        Execute unconstrained optimization.
        '''
        
        self.constraint_mode = False
        self.clipping_mode = False
        self.capacity = capacity
        print(f'--unconstrained mode:{self.constraint_mode}--clip mode:{self.clipping_mode}')
        self.tdens_best, self.pot_best, self.flux_best_norm, self.minCost, self.minCost_list, self.flux_best, self.min_g_list = dyn(self, capacity)

    def exec_constr(self, capacity):
        '''
        Execute constrained optimization.
        '''
        self.constraint_mode = True
        self.capacity = capacity
        print(f'--constrained mode:{self.constraint_mode}')
        print(f'capacity:{self.capacity[0]}')
        self.tdens_best, self.pot_best, self.flux_best_norm, self.minCost, self.minCost_list, self.flux_best, self.min_g_list = dyn(self,capacity)

    def exec_clip(self,capacity):
        '''
        Execute clipping operation.
        '''
        self.capacity = capacity
        self.clipping_mode = True
        print(f'--unconstrained mode:{self.constraint_mode}--clip mode:{self.clipping_mode}')
        self.constraint_mode = False

        if not self.constraint_mode and self.clipping_mode:
            self.tdens_best, self.pot_best, self.flux_best_norm, self.minCost, self.minCost_list, self.flux_best, self.min_g_list = dyn(self,capacity)

    def export_results(self):
        return self.tdens_best, self.pot_best, self.flux_best_norm, self.minCost, self.minCost_list, self.flux_best, self.min_g_list
