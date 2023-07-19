import os
import math
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt



class DDN:
    def __init__(self):
        print("DNN package")
    
    def readGeneName(self, filename):
        with open(filename, 'r') as file:
            genename = file.read().split('\n')
        while len(genename) > 0 and genename[-1] == '':
            genename.pop()
        return genename
    
    def readGeneData(self, filename):
        genedata = np.loadtxt(filename)
        genedata = genedata.transpose()
        return genedata
    
    def standardizeGeneData(self, genedata, scaler='rms', zero_mean=True):
        standarddata = np.zeros(genedata.shape)
        for i in range(genedata.shape[1]):
            u = np.mean(genedata[:, i]) if not zero_mean else 0
            if scaler == 'std':
                s = np.std(genedata[:, i])
            elif scaler == 'rms':
                s = np.sqrt(np.mean(np.square(genedata[:, i])))
            else:
                s = 1
            standarddata[:, i] = (genedata[:, i] - u) / s
        return standarddata
    
    def concatenateGeneData(self, controldata, casedata, method='diag'):
        if method == 'row':
            return np.concatenate((controldata, casedata), axis=0)
        elif method == 'col':
            return np.concatenate((controldata, casedata), axis=1)
        elif method == 'diag':
            return np.concatenate((np.concatenate((controldata, casedata * 0), axis=0), 
                                   np.concatenate((controldata * 0 ,casedata), axis=0)), axis=1)
        else:
            return []

    def solve2d(self, rho1, rho2, lambda1, lambda2):
        area_index = 0
        beta1 = 0
        beta2 = 0
        if (rho2 <= (rho1 + 2*lambda2) and rho2 >= (rho2 - 2*lambda2) and rho2 >= (2*lambda1 - rho1)):
            area_index = 1
            beta1 = (rho1 + rho2)/2 - lambda1
            beta2 = (rho1 + rho2)/2 - lambda1
        if (rho2 > (rho1 + 2*lambda2) and rho1 >= (lambda1 - lambda2)):
            area_index = 2
            beta1 = rho1 - lambda1 + lambda2
            beta2 = rho2 - lambda1 - lambda2
        if (rho1 < (lambda1 - lambda2) and rho1 >= -(lambda1 + lambda2) and rho2 >= (lambda1 + lambda2)):
            area_index = 3
            beta1 = 0
            beta2 = rho2 - lambda1 - lambda2
        if (rho1 < -(lambda1 + lambda2) and rho2 >= (lambda1 + lambda2)):
            area_index = 4
            beta1 = rho1 + lambda1 + lambda2
            beta2 = rho2 - lambda1 - lambda2
        if (rho1 < -(lambda1 + lambda2) and rho2 < (lambda1 + lambda2) and rho2 >= -(lambda1 + lambda2)):
            area_index = 5
            beta1 = rho1 + lambda1 + lambda2
            beta2 = 0
        if (rho2 < -(lambda1 - lambda2) and rho2 >= (rho1 + 2*lambda2)):
            area_index = 6
            beta1 = rho1 + lambda1 + lambda2
            beta2 = rho2 + lambda1 - lambda2
        if (rho2 >= (rho1 - 2*lambda2) and rho2 < (rho1 + 2*lambda2) and rho2 <= (-2*lambda1 - rho1)):
            area_index = 7
            beta1 = (rho1 + rho2)/2 + lambda1
            beta2 = (rho1 + rho2)/2 + lambda1
        if (rho2 < (rho1 - 2*lambda2) and rho1 <= -(lambda1 - lambda2)):
            area_index = 8
            beta1 = rho1 + lambda1 - lambda2
            beta2 = rho2 + lambda1 + lambda2
        if (rho1 <= (lambda1 + lambda2) and rho1 >= -(lambda1 - lambda2) and rho2 <= -(lambda1 + lambda2)):
            area_index = 9
            beta1 = 0
            beta2 = rho2 + lambda1 + lambda2
        if (rho1 > (lambda1 + lambda2) and rho2 <= -(lambda1 + lambda2)):
            area_index = 10
            beta1 = rho1 - lambda1 - lambda2
            beta2 = rho2 + lambda1 + lambda2
        if (rho2 > -(lambda1 + lambda2) and rho2 <= (lambda1 - lambda2) and rho1 >= (lambda1 + lambda2)):
            area_index = 11
            beta1 = rho1 - lambda1 - lambda2
            beta2 = 0
        if (rho2 > (lambda1 - lambda2) and rho2 < (rho1 - 2*lambda2)):
            area_index = 12
            beta1 = rho1 - lambda1 - lambda2
            beta2 = rho2 - lambda1 + lambda2
        return [beta1, beta2]
    
    def bcd(self, y, X, lambda1, lambda2, n1=1, n2=1, threshold=1e-6):
        if (X.shape[1] == 0 or X.shape[1] % 2 == 1):
            return []
        p = X.shape[1] // 2
        beta = np.zeros(2 * p)
        if (p == 1):
            rho1 = np.sum(y * X[:, 0]) / n1
            rho2 = np.sum(y * X[:, 1]) / n2
            beta2d = self.solve2d(rho1, rho2, lambda1, lambda2)
            beta[0] = beta2d[0]
            beta[1] = beta2d[1]
            return beta
        else:
            while (True):
                beta_old = np.copy(beta)
                for k in range(p):
                    x1 = X[:, k]
                    x2 = X[:, k + p]
                    idx = [i for i in range(2 * p) if i not in (k, k + p)]
                    y_residual = y - np.dot(X[:, idx], beta[idx])
                    rho1 = np.sum(y_residual * x1) / n1
                    rho2 = np.sum(y_residual * x2) / n2
                    beta2d = self.solve2d(rho1, rho2, lambda1, lambda2)
                    beta[k] = beta2d[0]
                    beta[k + p] = beta2d[1]
                if (np.mean(np.abs(beta - beta_old)) < threshold):
                    break
            return beta

    def generateNetworks(self, case_data, control_data, genename, lambda1, lambda2, threshold=1e-6):
        p = control_data.shape[1]
        n1 = control_data.shape[0]
        n2 = case_data.shape[0]

        common_edges = {}
        differential_edges = {}
        
        for gene in tqdm(range(p)):
            y = self.concatenateGeneData(control_data[:, gene], case_data[:, gene], method='row')
            idx = [i for i in range(p) if i != gene]
            X = self.concatenateGeneData(control_data[:, idx], case_data[:, idx], method='diag')
            beta = self.bcd(y, X, lambda1, lambda2, n1, n2, threshold)

            beta1 = list(beta[0 : gene]) + [0] + list(beta[gene : p - 1])
            beta2 = list(beta[p - 1 : gene + p - 1]) + [0] + list(beta[gene + p - 1 : 2 * p - 2])

            common_conditions = [genename[i] for i in range(p) if (beta1[i] != 0 and beta2[i] != 0) or (beta1[i] == 0 and beta2[i] == 0)]
            weight_common = [beta1[i] for i in range(p) if (beta1[i] != 0 and beta2[i] != 0) or (beta1[i] == 0 and beta2[i] == 0)]
            
            condition1 = [genename[i] for i in range(p) if beta1[i] != 0 and beta2[i] == 0]
            condition2 = [genename[i] for i in range(p) if beta2[i] != 0 and beta1[i] == 0]
            weight1 = [beta1[i] for i in range(p) if beta1[i] != 0 and beta2[i] == 0]
            weight2 = [beta2[i] for i in range(p) if beta2[i] != 0 and beta1[i] == 0]
            
            for neighbor, weight in zip(common_conditions, weight_common):
                tuple_common_edge = (min(genename[gene], neighbor), max(genename[gene], neighbor), 'common')
                common_edges.setdefault(tuple_common_edge, 0.0)
                common_edges[tuple_common_edge] += weight

            for neighbor, weight in zip(condition1, weight1):
                tuple_diff_edge1 = (min(genename[gene], neighbor), max(genename[gene], neighbor), 'condition1')
                differential_edges.setdefault(tuple_diff_edge1, 0.0)
                differential_edges[tuple_diff_edge1] += weight
            
            for neighbor, weight in zip(condition2, weight2):
                tuple_diff_edge2 = (min(genename[gene], neighbor), max(genename[gene], neighbor), 'condition2')
                differential_edges.setdefault(tuple_diff_edge2, 0.0)
                differential_edges[tuple_diff_edge2] += weight
                
        common_edges = sorted([k + tuple([v]) for k, v in common_edges.items()])
        differential_edges = sorted([k + tuple([v]) for k, v in differential_edges.items()])

        return common_edges, differential_edges
    
    def plotDifferentialNetwork(self, diffedges, maxalpha=1.0, minalpha=0.2, node_size_factor=100):
        G = nx.Graph()
        color_condition = {'condition1': [1, 0, 0], 'condition2': [0, 0, 1], 'common': [0, 1, 0]}
        maxbeta = max([beta for _, _, _, beta in diffedges])

        for gene1, gene2, condition, beta in diffedges:
            if condition in color_condition:
                alpha = beta / maxbeta * (maxalpha - minalpha) + minalpha
                color = list(1 - (1 - np.array(color_condition[condition])) * alpha)
                G.add_edge(gene1, gene2, color=color)

        pos = nx.circular_layout(G)
        edges = G.edges()
        edge_color = [G[u][v]['color'] for u, v in edges]
        node_size = [d * 200 for n, d in G.degree()]

        fig, ax = plt.subplots(figsize=(6, 6))
        nx.draw(G, pos=pos, node_color='lightblue', node_size=node_size, \
                edgelist=edges, edge_color=edge_color, width=3.5, \
                with_labels=True, font_size=16, font_weight='bold', font_color='magenta')

        ax.set_xlim((-1.2, +1.2))
        ax.set_ylim(ax.get_xlim())
        ax.set_title(''.join([
            'DDN Network\n', 
            'condition1: red edge\n', 
            'condition2: blue edge\n',
            'common: green edge\n'
        ]))
        
        plt.show()

#     def printDifferentialNetwork(self, diffedges, filename=''):
#         if len(filename) == 0:
#             for gene, neighbor, condition, weight in diffedges:
#                 print(f"{gene},{neighbor},{condition},{weight}")
#         else:
#             with open(filename, 'w') as file:
#                 for gene, neighbor, condition, weight in diffedges:
#                     file.write(f"{gene},{neighbor},{condition},{weight}\r")

    def DDNPipline(self, case_data_file, control_data_file, gene_name_file, output_file='', lambda1=0.10, lambda2=0.03):
        casedata = self.readGeneData(case_data_file)
        controldata = self.readGeneData(control_data_file)
        genename = self.readGeneName(gene_name_file)

        assert(casedata.shape[1] == controldata.shape[1])

        case_standard = self.standardizeGeneData(casedata)
        control_standard = self.standardizeGeneData(controldata)

        common_edges, differential_edges = self.generateNetworks(case_standard, control_standard, genename, lambda1, lambda2)

        
        #self.printDifferentialNetwork(common_edges, output_file)
        #self.printDifferentialNetwork(differential_edges, output_file)

        if not common_edges and not differential_edges:
            print(f"No common or differential edges found for node {gene_name_file}. Moving to the next node.")
            return None

        self.plotDifferentialNetwork(common_edges, node_size_factor=2)
        self.plotDifferentialNetwork(differential_edges, node_size_factor=100)

        
        print(common_edges)
        print(differential_edges)
        return common_edges, differential_edges


