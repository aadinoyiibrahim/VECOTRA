"""
VECOTRA : VElocity Constrained Optinmal TRAnsport
    Constributors:
                Abdullahi Ibrahim
                Caterina De Bacco
"""

from libpysal import weights, examples
from libpysal.cg import voronoi_frames
from scipy.spatial import distance, Delaunay
import geopandas
import networkx as nx
import numpy as np
import pandas as pd
from itertools import combinations
from math import  sqrt
import random

# ===================
''' 1. generate planar '''
def planar_graph_net(self):
    prng = np.random.RandomState(self.seedG)
    G0 = None
    subN = None
    L_max=None
    L_min = None
    domain=(0, 0, 1, 1)
    metric=None
    connected_component=True
    if G0 is None:
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.N))
        (xmin, ymin, xmax, ymax) = domain
        # Each node gets a uniformly random position in the given rectangle.
        pos = {v: (prng.uniform(xmin, xmax), prng.uniform(ymin, ymax)) for v in G}
        nx.set_node_attributes(G, pos, "pos")
    else:
        print('using G')
        G = nx.Graph()
        G.add_nodes_from(G0.nodes(data=True))
        nnode = G.number_of_nodes()
        pos = nx.get_node_attributes(G, "pos")

    if subN is None: subN = self.N

    if subN < self.N:
        random.seed(self.seedG)
        node2remove = random.sample(list(G.nodes()),k = self.N-subN)
        G.remove_nodes_from(node2remove)

    ### Extracts planar graph
    nodes = list(G.nodes())
    coordinates = np.array([(pos[n][0],pos[n][1]) for n in list(G.nodes())])
    cells, generators = voronoi_frames(coordinates, clip="convex hull")
    delaunay = weights.Rook.from_dataframe(cells)
    G = delaunay.to_networkx()
    G = nx.relabel_nodes(G, { i: nodes[i] for i in range(G.number_of_nodes())})
    positions = { n: coordinates[i] for i,n in enumerate(list(G.nodes())) }
    nx.set_node_attributes(G, positions, "pos")

    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = euclidean

    if L_max is None:
        L_max = max(metric(x, y) for x, y in combinations(positions.values(), 2))
    if L_min is None:
        L_min = 0

    def dist(u, v):
        return metric(positions[u], positions[v])

    edges2remove = [e for e in list(G.edges()) if np.logical_or(dist(*e) > L_max,dist(*e) < L_min ) == True]
    G.remove_edges_from(edges2remove)

    if connected_component == True:
        Gc = max(nx.connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

    return G

def file2graph(edges_file_name,sep = ' ',weight_inter = 1, w0 = 1 ):
    '''
    Generate graph from edgelist of rows:
    layerId sourceId targetId weight
    '''

    edges_file = pd.read_csv(edges_file_name, names=['layer', 'fromNode', 'toNode', 'weight'], sep=sep)

    nodes_inter = extract_node_layer(edges_file) #dict(nodeID)

    layers = list(np.unique(edges_file.layer))
    nlayer = len(layers)
    
    '''
    Build graph:
    - single-layer network
    - add dummy nodes for nodes that belong to more than one layer
    - add dummy supernode connecting them
    - inter-layer edges are between dummy nodes
    '''

    g = nx.Graph()
    for n,row in edges_file.iterrows():
        l = row[0]
        lid = layers.index(l)
        u0,v0 = str(row[1]), str(row[2])
        w = row[3]

        u = str(u0) + '_' + str(lid)
        v = str(v0) + '_' + str(lid)

        g.add_edge(u,v, weight = w, etype= lid )

    '''
    Add inter-layer edges
    '''
    for n in nodes_inter:
        if len(nodes_inter[n]) > 1:
            g.add_node(str(n), ntype='super')
            for l in nodes_inter[n]:
                u = str(n) + '_' + str(l)
                g.add_edge(str(n),u,weight=w0,etype='inter-super')
                g.nodes[u]['ntype'] = 'inter'
                # for m in nodes_inter[n]:
                #     if l != m :
                #         v = str(n) + '_' + str(m)
                #         g.add_edge(u,v,weight=weight_inter,etype='inter')
        else:
            u = str(n) + '_' + str(nodes_inter[n][0])
            g.nodes[u]['ntype'] = 'intra'
    

    nodes = list(g.nodes()) #list of stations
    nnode = len(nodes)

    '''
    Relabel nodes so that nodeId is the same as node name
    '''
    nodeName2Id = {}
    nodeId2Name = {}
    for i,n in enumerate(nodes): 
        nodeName2Id[n] = i
        nodeId2Name[i] = n
        if '_' in n: # inter-layer nodes
            n0 = n.split('_')[0]
            nodeName2Id[n0] = i

    g = nx.relabel_nodes(g, nodeName2Id)
    
    nodes = list(np.arange(nnode))
        
    nedge = g.number_of_edges() 

    return g, nnode, nedge, nodes, nodes_inter, nodeName2Id, nodeId2Name

def from_nx_graph2coord(list_G, nodes0,cols = ['nodeID', 'name', 'nodeLat', 'nodeLong'],outfile = 'coord_tmp.csv',mapping = None, nodes_inter = None):
    data = []
    nodes = []
    for l in range(len(list_G)):
        for n0_int in list(list_G[l].nodes()):
            n0 = str(n0_int)
            #for m in nodes_inter[n0]:
            n_name = str(n0) # + '_' + str(m)
            if mapping is not None:
                n = mapping[str(n_name)]
            else:
                n = n0
            if n_name not in nodes:
                i = nodes0.index(n)
                data.append([i,n_name,list_G[l].nodes[n0_int]['pos'][0],list_G[l].nodes[n0_int]['pos'][1]])
                nodes.append(n_name)
            '''
            if len(nodes_inter[n0]) > 1:
                n_name = str(n0)
                if n_name not in nodes:
                    if mapping is not None:
                        n = mapping[str(n_name)]
                    else:
                        n = n0
                    i = nodes0.index(n)
                    data.append([i,n_name,list_G[l].nodes[n0_int]['pos'][0],list_G[l].nodes[n0_int]['pos'][1]])
                    nodes.append(n_name)'''

    df = pd.DataFrame(data,columns = cols)
    if outfile is not None:
        df.to_csv(outfile,index=False,header=0)
    return df

def extract_node_layer(df):
    nodes_inter = {}
    for n0,g in df.groupby(by=['fromNode']):
        n = str(n0)
        layer_i = set(g['layer'].unique())
        if n in nodes_inter:
            nodes_inter[n] = layer_i.union(nodes_inter[n])
        else:
            nodes_inter[n] = set(layer_i)
    for n0,g in df.groupby(by=['toNode']):
        n = str(n0)
        layer_i = set(g['layer'].unique())
        if n in nodes_inter:
            nodes_inter[n] = layer_i.union(nodes_inter[n])
        else:
            nodes_inter[n] = set(layer_i)

    for n in nodes_inter:
        nodes_inter[n] = list(nodes_inter[n])

    return nodes_inter 

def euclidean(x, y):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))
# ================

# -------------
 # 2. generate forcing

def forcing_generate(G, p = 0., weigth = 10, pos = None, pos_center_of_mass = (0.5,0.5), seed = 10):
    """G is the graph in the first layer"""
    
    prng = np.random.RandomState(seed)
    
    n_center = find_central_nodes(G, pos = None, pos_center_of_mass = None)
    print('n_center:',n_center)
    nodes = set(G.nodes()) - {n_center} # all nodes except central one
    forcing = []
    for source in nodes:
        if source != n_center:
            r = prng.rand()
            if r < p: # rewire: extract a random node
                sink = random.choice(list(nodes - {source}))
                forcing.append([source,sink,weigth])
            else:
                forcing.append([source,n_center,weigth])
    
    return forcing

def find_central_nodes(G,pos = None,pos_center_of_mass = None):
    
    if pos is None:
        pos = nx.get_node_attributes(G,'pos')
    if pos_center_of_mass is None:
        positions = np.array(list(pos.values()))
        pos_center_of_mass = np.mean(positions,axis = 0)
    nodes = list(G.nodes)
    
    distances = [euclidean(pos[n],pos_center_of_mass) for n in nodes]
    nid = np.argmin(distances)
    n = nodes[nid]
    return n

def forcing_importing_from_file( input_path, nodes,nodes_inter,nodeName2Id = None,sep = ' ',header=True,source='source',sink='sink'):
    """import forcing from last run
    source sink weight
    0 1 10
    """
    print("forcing: imported from file")

    rhs_df = pd.read_csv(input_path,sep=sep,header=header)
    comm_list = list(rhs_df[source].unique().astype('str')) # keep only comm with g^i > 0
    sink_list = list(rhs_df[sink].unique().astype('str')) # keep only comm with h^i != 0
    
    transit_list = list(set(nodes).difference(set(comm_list).union(set(sink_list))))
    print('Ncom:',len(comm_list),' Ntransit:',len(transit_list))

    nnode = len(nodes)
    ncomm = len(comm_list)

    forcing = np.zeros((ncomm, nnode))
    for idx,row in rhs_df.iterrows():
        source_i = str(int(row[0]))
        sink_i = str(int(row[1]))
        g_i = row[2]
        idx_com = comm_list.index(source_i)

        if len(nodes_inter[source_i]) > 1: #super node
            if nodeName2Id is not None:
                i = nodeName2Id[source_i]
            else:
                i = source_i
            forcing[idx_com][i] += g_i
        else:
            if nodeName2Id is not None:
                i = nodeName2Id[source_i + '_' + str(nodes_inter[source_i][0])]
            else:
                i = source_i + '_' + str(nodes_inter[source_i][0])
            forcing[idx_com][i] += g_i

        if len(nodes_inter[sink_i]) > 1: # super node
            if nodeName2Id is not None:
                j = nodeName2Id[sink_i]
            else:
                j = sink_i
            forcing[idx_com][j] -= g_i
        else:
            if nodeName2Id is not None:
                j = nodeName2Id[sink_i + '_' + str(nodes_inter[sink_i][0])]
            else:
                j = sink_i + '_' + str(nodes_inter[sink_i][0])
            forcing[idx_com][j] -= g_i
      
    return forcing, comm_list, transit_list
# -------------
def from_nx_graphs2df(list_G,cols = ['layer', 'fromNode', 'toNode', 'weight'],outfile = 'adjacency_tmp.csv'):
    data = []
    for l in range(len(list_G)):
        for i,j in list(list_G[l].edges()):
            data.append([l,i,j,1])
            
    df = pd.DataFrame(data,columns = cols)
    if outfile is not None:
        df.to_csv(outfile,index=False,header=0)
    return df
# ===========
def coord_importing(coord_file_name, g, nid='nodeID',nlat='nodeLat',nlon='nodeLong',sep=' '):
    """coordinates imported from file"""

    print("coordinates: imported")

    '''
    Build dictionary of (lat,lon)
    '''
    df_coord = pd.read_csv(coord_file_name, names=['nodeID', 'name', 'nodeLat', 'nodeLong'], sep = sep)

    pos_lat = df_coord.set_index(nid).to_dict()[nlat]
    pos_lon = df_coord.set_index(nid).to_dict()[nlon]
    pos = dict()
    for i in pos_lat: pos[i] = (pos_lat[i],pos_lon[i])   

    for i in list(g.nodes):
        if i in pos:
            g.nodes[i]["pos"] = pos[i]
        else:
            print('node ',i, 'does not have coordinates!')


# ===========
def compute_euclidean(g):
    length = np.zeros(g.number_of_edges())
    for i, edge in enumerate(g.edges()):
        length[i] = distance.euclidean(g.nodes[edge[0]]["pos"], g.nodes[edge[1]]["pos"])
    return length










