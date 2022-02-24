"""
    AUTHOR : IL-SEUNG JANG

    INITIAL VERSION               LAST UPDATE
    2021.11.07                    2022.02.24
"""

"""
    INFO:
        - This has been developed on `SageMath 9.4`.
    
    DESCRIPTION:
        - Compute the `q`-character of KR modules in all untwisted affine types 
          by using Hernandez-Leclerc's clsuter algebra approach to them.

    REFERENCE:
        [FM01] Edward Frenkel and Evgeny Mukhin, 
               Combinatorics of {$q$}-characters of finite-dimensional representations of quantum affine algebras, 
               Comm. Math. Phys. {216} (2001), no.~1, 23--57.
        
        [HL16] David Hernandez and Bernard Leclerc, 
               A cluster algebra approach to {$q$}-characters of {K}irillov-{R}eshetikhin modules, 
               J. Eur. Math. Soc. (JEMS) {18} (2016), no.~5, 1113--1159.

    CODE REFERENCE:
        [MS11] G. Musiker, C. Stump, 
               A Compendium on the cluster algebra and quiver package in Sage, preprint (2011), 
               arXiv:1102.4844 (Sem. Lothar. Combin. 67, 67 pages, 2011).

    PREREQUISITES:
        - Prerequisites for the optional package `slabbe` are `dot2tex`, `glucose`, `latte_int`.
        - Prerequisites in system packages: `graphviz` `imagemagick` and `pdf2svg``
"""
import os
import numpy
import shutil
from copy import deepcopy

from sage.structure.sage_object import SageObject
from sage.combinat.root_system.cartan_type import CartanType
from sage.data_structures.blas_dict import add as blas_dict_add
from sage.misc.temporary_file import tmp_filename # to deal with the files in SAGE_TMP
from sage.misc.misc import SAGE_TMP # SAGE_TMP : sage temporary folder

# ! Import optional packages to display results
try:
    from tqdm import *
    tqdm_check = True
except ImportError as e:
    tqdm_check = False
    pass

try:
    # from slabbe import *
    from slabbe import *
    slabbe_check = True
except ImportError as e:
    slabbe_check = False
    pass

# ! Import optional packages to deal with images
try:
    from PIL import Image, ImageFont, ImageDraw
    PIL_check = True
except ImportError as e:
    PIL_check = False
    pass

try:
    from IPython.display import display
    display_check = True
except ImportError as e:
    display_check = False
    pass

# !-----------------------------------------------------------------
"""
! STRUCTURE OF THIS CODE:

    * [functions] 
        * W_minus_initial (for top row)
			└──  W_minus  (= vertex set of `G^-`)
                
        * (classes) Monomial (for single monomial of variable `Y_{i,a}`),
            └── qCharacter (for Laurent polynomial of `Y_{i,a}`) 
				└── (function) convert
								└── substitute the cluster variables to the monomials `z`'s of variable `Y_{i,a}`
                └── (function) inverse_of_monomial
                                └──  return the inverse of `single` monomial of variable `Y_{i,a}`
        * compare
            └── compare two matrices
            
        * naming
            └── to rename the resulting files and move them in the folder located in the source file.
            
    * [classes] 
        * G_minus (= the quiver `G^-`) ; main part of this code
			│
		    ├── self._non_shift_vertex_set : return the vertex set V,
			│
			├── self._vertex_set : return the vertex set W,
			│
			├── self._cartan_type : return its Cartan type,
			│    
			├── self._cartan_matrix : return its Cartan matrix,
			│    
			├── self._B : return the symmetrized matrix of `Gm._cartan_matrix`,
			│    
			├── self._matrix : return its adjacent matrix of quiver `Gm`,
			│    
			├── self._cluster : return its cluster variables determined by `Gm._matrix`,
			│    
			├── self._count_mutations : return the number of mutations currently, 
			│    
			├── self.quiver : return its quiver (by using `cluster algebra` package),
			│    
			├── self.mutation(`vertex`) : apply mutation at `vertex`,
			│    
			├── self.initial : initialize the data related to quiver mutation
			│    
			├── self.mutations(`sequence`) : apply the `sequence` of mutations,
			│    
			├── self.mu : return its (standard) mutation sequence,
			│    
			├── self.generator_monomials : return the monomials z_{i,r}
			│    
			└── self.change_of_variables : return the Laurent polynomial of 
						    `Y` of a cluster variable by change of variables
"""
# !-----------------------------------------------------------------


def W_minus_initial(ct):
    """
        INPUT:
            `ct` : a Cartan type (of finite type),

        DESCRIPTION:
            Return top vertex in the quiver `G^-` in type `ct`.
    """
    I = CartanType(ct).index_set()
    rank = CartanType(ct).rank()

    W = {}
    if ct[0] == 'A':
        for idx in I:
            if idx % 2 == 0:
                W[idx] = {idx-1:(idx,-1)}
            if idx % 2 == 1:
                W[idx] = {idx-1:(idx,-2)}

    elif ct[0] == 'D':
        for idx in I:
            if rank % 2 == 0:
                if idx != rank:
                    if idx % 2 == 0:
                        W[idx] = {idx-1:(idx,-1)}
                    if idx % 2 == 1:
                        W[idx] = {idx-1:(idx,-2)}
                elif idx == rank:
                    W[idx] = {idx-1:(idx, W[idx-1][idx-2][1])}

            if rank % 2 == 1:
                if idx != rank:
                    if idx % 2 == 0:
                        W[idx] = {idx-1:(idx,-2)}
                    if idx % 2 == 1:
                        W[idx] = {idx-1:(idx,-1)}
                elif idx == rank:
                    W[idx] = {idx-1:(idx, W[idx-1][idx-2][1])}
    
    elif ct[0] == 'E':
        if ct[1] == 6:
            """
                E6 Dynkin diagram:
                            O 2
                            |
                    O---O---O---O---O
                    1   3   4   5   6  
            """
            W = {1:{0:(1,-1)}, 2:{1:(2,-2)}, 3:{2:(3,-2)}, 4:{3:(4,-1)}, 5:{4:(5,-2)}, 6:{5:(6,-1)}}

        elif ct[1] == 7:
            """
                E7 Dynkin diagram:
                        O 2
                        |
                O---O---O---O---O---O
                1   3   4   5   6   7 
            """
            W = {1:{0:(1,-1)}, 2:{1:(2,-2)}, 3:{2:(3,-2)}, 4:{3:(4,-1)}, 5:{4:(5,-2)}, 6:{5:(6,-1)}, 7:{6:(7,-2)}}

        elif ct[1] == 8:
            """
                E8 Dynkin diagram:
                        O 2
                        |
                O---O---O---O---O---O---O
                1   3   4   5   6   7   8
            """
            W = {1:{0:(1,-1)}, 2:{1:(2,-2)}, 3:{2:(3,-2)}, 4:{3:(4,-1)}, 5:{4:(5,-2)}, 6:{5:(6,-1)}, 7:{6:(7,-2)}, 8:{7:(8,-1)}}

    elif ct[0] == 'C':
        for idx in I:
            if rank % 2 == 1:
                if idx != rank:
                    if idx % 2 == 0:
                        W[idx] = {idx-1:(idx,-2)}
                    if idx % 2 == 1:
                        W[idx] = {idx-1:(idx,-1)}
                if idx == rank:
                    W[idx] = {rank-1:(idx, W[idx-1][idx-2][1])}
            
            if rank % 2 == 0:
                if idx != rank:
                    if idx % 2 == 0:
                        W[idx] = {idx-1:(idx,-1)}
                    if idx % 2 == 1:
                        W[idx] = {idx-1:(idx,-2)}
                if idx == rank:
                    W[idx] = {rank-1:(idx, W[idx-1][idx-2][1])}

    elif ct[0] == 'B':
        for idx in I:
            if rank % 2 == 0:
                if idx % 2 == 0:
                    W[idx] = {idx-1:(idx,-5)}
                if idx % 2 == 1:
                    W[idx] = {idx-1:(idx,-3)}

            if rank % 2 == 1:
                if idx % 2 == 0:
                    W[idx] = {idx-1:(idx,-3)}
                if idx % 2 == 1:
                    W[idx] = {idx-1:(idx,-5)}

        W[rank][rank-1] = (rank, -1)


    elif ct[0] == 'F':
        """
        (for record) In [HL16], the Cartan data for type F is as follows:

                    Dynkin diagram:  O---O=<=O---O
                                      1   2   3   4 
                    
                    Cartan matrix:
                    C = matrix([
                        [ 2, -1,  0,  0],
                        [-1,  2, -2,  0],
                        [ 0, -1,  2, -1],
                        [ 0,  0, -1,  2]])

                    Symmetrizer :  D = {1:1,2:1,3:2,4:2}

        """
        W = {1:{0:(1,-1)}, 2:{1:(2,-2)}, 3:{2:(3,-2)}, 4:{3:(4,-2)}}

    elif ct[0] == 'G':
        """
        (for record) In [HL16], the Cartan data for type F is as follows:

                Dynkin diagram:
                                      3
                                    O=>=O
                                    1   2
                Cartan matrix:
                C = matrix([
                        [ 2, -1],
                        [-3,  2]])

                Symmetrizer :  D = {1:3, 2:1}
        """
        W = {1:{0:(1,-4)}, 2:{1:(2,-1)}}

    return dict(sorted(W.items()))


def W_minus(ct, m, shift=True):
    """
        INPUT:
            `ct` : a Cartan type (of finite type),
            `m` : the maximal length of column

        DESCRIPTION:
            Return a subset of `W^-` (type: `dict`) whose quiver has height less 
            than `m`. (it will be used to generate a subquiver of the quiver `G^-`.)
        
        DATA STRUCTURE:
        TYPE `dict`
                   { ...,  j : {...        ,   k :      (j,s) , ...}, ... }
                           |                   |           |
                      index                    numbering   vertex

        EXAMPLE:
            sage: W_minus(['A',3], 3)

            {1: {(1, -1): 0, (1, -3): 3, (1, -5): 6},
             2: {(2, 0): 1, (2, -2): 4, (2, -4): 7},
             3: {(3, -1): 2, (3, -3): 5, (3, -5): 8}}

            The above vertex set corresponds to the following quiver:

            idx:    1             2            3

                (1,-1):0 ---- (2, 0):1 ---- (3,-1):2
                   |     ^-_      |   _-^      |
                (1,-3):3 ---- (2,-2):4 ---- (3,-3):5
                   |     _-^      |    ^-_      |
                (1,-5):6 ---- (2,-4):7 ---- (3,-5):8

            Here, (i,j):k denote the `k`th vertex `(i,j)`.
    """
    I = CartanType(ct).index_set()
    rank = CartanType(ct).rank()

    if ct[0] != 'F' and ct[0] != 'G':
        CM = CartanMatrix(ct) # Cartan matrix of type `ct`
        B = CM.symmetrized_matrix() # the symmetric matrix `B = DC`
        D = CM.symmetrizer()

    # ! In [HL16], the convention for types F and G is reversed, so transpose the Cartan matrix.
    elif ct[0] == 'F': 
        CM = CartanMatrix(ct).transpose()
        B = matrix([
            [ 2, -1,  0,  0],
            [-1,  2, -2,  0],
            [ 0, -2,  4, -2],
            [ 0,  0, -2,  4]])
        D = {1:1,2:1,3:2,4:2}
    elif ct[0] == 'G':
        CM = CartanMatrix(ct).transpose()
        B = matrix([
            [6,-3],
            [-3,2]
        ])
        D = {1:3, 2:1}
    
    # check whether m is positive
    assert (m > 0), "The last parameter must be positive."

    V = W_minus_initial(ct) # ! initialize the vertex set

    for idx in I:
        for k in range(m):
            V[idx][rank*k+idx-1] = (idx, V[idx][idx-1][1]-k*CM[idx-1][idx-1])

    if shift == True:
        for idx in I:
            for label, vertex in V[idx].items():
                V[idx][label] = (idx, vertex[1]+D[idx])

    if ct[0] == 'B':
        for idx in I:
            if rank % 2 == 1 and idx % 2 == 1:
                if idx != rank:
                    for label, vertex in V[idx].items():
                        V[idx][label] = (idx, vertex[1]+CM[idx-1][idx-1])

            if rank % 2 == 0 and idx % 2 == 0:
                if idx != rank:
                    for label, vertex in V[idx].items():
                        V[idx][label] = (idx, vertex[1]+CM[idx-1][idx-1])

    return V

# ! Main part of this code
class G_minus(SageObject):
    """
        INPUT:
            `ct` : a Cartan type (of finite type),
            `m` : the maximal length of column

        DESCRIPTION:
            - G_minus(ct, max) : Compute a sub-quiver of the quiver `G^-`.
    """
    def __init__(self, ct, m, display=False):
        self._vertex_set = W_minus(ct, m)
        self._non_shift_vertex_set = W_minus(ct, m, shift=False)
        self._max = m # maximal size of G-
        self._cartan_type = CartanType(ct)

        if display==True:
            self._display = True
        else:
            self._display = False

        if self._cartan_type[0] != 'F' and self._cartan_type[0] != 'G':
            self._cartan_matrix = CartanMatrix(ct)
            self._B = self._cartan_matrix.symmetrized_matrix()

        elif self._cartan_type[0] == 'F':
            self._cartan_matrix = CartanMatrix(ct).transpose()
            self._B = matrix([ [ 2, -1,  0,  0], [-1,  2, -2,  0], [ 0, -2,  4, -2],[ 0,  0, -2,  4]])
        
        elif self._cartan_type[0] == 'G':
            self._cartan_matrix = CartanMatrix(ct).transpose()
            self._B = matrix([[6,-3], [-3,2]])
            # D = {1:3, 2:1}

        if self._cartan_type[0] in ['A', 'D', 'E']:
            self._t = 1
        elif self._cartan_type[0] in ['B','C', 'F']:
            self._t = 2
        elif self._cartan_type[0] in ['G']:
            self._t = 3
        

        self._initial_matrix = self.matrix()
        self._matrix = self.matrix()

        # self._initial_cluster_algebra = ClusterSeed(self._initial_matrix)
        self._cluster_algebra = ClusterSeed(self.matrix())

        # self._initial_cluster = self._initial_cluster_algebra.cluster()
        self._cluster = self._cluster_algebra.cluster()

        mutation_count = {}
        for idx in self._cartan_type.index_set():
            for label, vertex in self._vertex_set[idx].items():
                mutation_count[label] = {vertex:[0,[]]}
        
        self._initial_count_mutations = mutation_count
        self._count_mutations = mutation_count
        self._order_mutation = 0


    def __eq__(self, other):
        return self._maxtix == other._matrix and self._vertex_set == other._vertex_set


    def __ne__(self, other):
        return not (self == other)


    def _repr_(self):
        l = len(self.matrix().columns())
        
        if self._display == True:
            print("The quiver G^- of type "+str(self._cartan_type)+" with vertex set:"+"\n")
            self.show_vertex()

            print("and adjacency "+str(l)+" x "+str(l)+" matrix "+"( max = "+str(self._max)+" )"+":"+"\n")
            return str(self.matrix())

        elif self._display == False:
            return "The quiver G^- of type "+str(self._cartan_type)+" with adjacency "+str(l)+" x "+str(l)+" ( max = "+str(self._max)+" )."


    def matrix(self):
        """
            DESCRIPTION:
                Return the matrix of `self` determined by `self._non_shift_vertex_set`
        """
        B = self._B
        V = self._non_shift_vertex_set
        N = 0
        
        for idx in V.keys():
            N += len(V[idx])

        M = matrix(N) # N x N empty matrix

        for idx1 in V.keys():
            for idx2 in V.keys():
                for la1 in V[idx1].keys():
                    for la2 in V[idx2].keys():
                        if B[idx1-1][idx2-1] !=0 and V[idx2][la2][1] == V[idx1][la1][1] + B[idx1-1][idx2-1]:
                            M[la1, la2] = 1
                            M[la2, la1] = -1

        return M


    def quiver(self, tikz=False, save=False, scale=2, label=None, remove_SAGE_TMP=False):
        """
            INPUT:
                `tikz` (optional, default = False) : Display the quiver in tikz style based on the optional package `slabbe`
                    - 'save'(optional, default = False) : Save the quiver of `self`
                (https://www.labri.fr/perso/slabbe/docs/0.6.3/tikz_picture.html)

                `save` (optional, default = False) : save the result
                'scale' (optional, default = 2) : scale of figure
                `label` (optional, default = None) : distinguish the vertex with `label`
                `remove_SAGE_TMP` (optional, default = False) : initial the folder in `SAGE_TMP`

            DESCRIPTION:
                Return the quiver (type : `digraph`) of `self` at current step
        """
        ct = self._cartan_type
        rank = ct.rank()
        Adj = self._matrix
        I = self._cartan_type.index_set()
        W = self._vertex_set
        vertex = {}
        edges = []
        pos = {}

        for idx in I:
            vertex.update(W[idx])

        lr = len(Adj.rows())
        lc = len(Adj.columns())

        if ct[0] in ['A','D','E']:
            for value in vertex.values():
                pos[value] = value

        elif ct[0] == 'B':
            for value in vertex.values():
                if value[0] == rank:
                    pos[value] = value
                else:
                    if value[0] % 2 == 1:
                        if value[1] % 4 == 3:
                            pos[value] = (2*rank-value[0], value[1])
                        elif value[1] % 4 == 1:
                            pos[value] = value
                    
                    elif value[0] % 2 == 0:
                        if value[1] % 4 == 3:
                            pos[value] = value
                        elif value[1] % 4 == 1:
                            pos[value] = (2*rank-value[0], value[1])


        elif ct[0] == 'C':
            for value in vertex.values():
                if value[0] != rank:
                    pos[value] = value
                else:
                    if value[1] % 4 == 0:
                        pos[value] = value
                    elif value[1] % 4 == 2:
                        pos[value] = (value[0]+1, value[1])

        elif ct[0] == 'F':
            for value in vertex.values():
                if value[0] not in [3, 4]:
                    pos[value] = value
                else:
                    if value[0] == 3:
                        if value[1] % 4 == 0:
                            pos[value] = value
                        elif value[1] % 4 == 2:
                            pos[value] = (value[0]+1, value[1])
                    elif value[0] == 4:
                        if value[1] % 4 == 0:
                            pos[value] = (value[0]+1, value[1])
                        elif value[1] % 4 == 2:
                            pos[value] = (value[0]+2, value[1])

        elif ct[0] == 'G':
            for value in vertex.values():
                if value[0] == 2:
                    pos[value] = value
                elif value[0] == 1:
                    if value[1] % 6 == 5:
                        pos[value] = (value[0]+2, value[1])
                    elif value[1] % 6 == 3:
                        pos[value] = (value[0], value[1])
                    elif value[1] % 6 == 1:
                        pos[value] = (value[0]+3, value[1])

        for i in range(lc): # row index
            for j in range(lr): # column index
                if Adj[i,j] == 1:
                    edges.append((vertex[i], vertex[j],''))
                if Adj[i,j] == -1:
                    edges.append((vertex[j], vertex[i],''))


        kwds = dict(format='list_of_edges')
        G = DiGraph(edges, **kwds)
        G.set_pos(pos)

        if label != None:
            relabel_list = {}
            for idx in I:
                for key in self._vertex_set[idx]:
                    if key == label:
                        relabel_list[self._vertex_set[idx][key]] = '$\\boxed{'+str(self._vertex_set[idx][key])+'}$'
                    else:
                        relabel_list[self._vertex_set[idx][key]] = str(self._vertex_set[idx][key])

            G.relabel(relabel_list)

        if tikz == False:
            return G

        elif tikz == True:
            if save == True:
                if slabbe_check == True:
                    foldername = "Result"+"_"+str(self._cartan_type[0])+str(self._cartan_type[1])+"_"+str(self._max)
                    filename = tmp_filename('quiver'+str(self._cartan_type)+"_"+"max["+str(self._max)+"]_"+str(self._order_mutation)+"_",'.png')
                    if label == None:
                        self.quiver(tikz=tikz, scale=scale).png(filename, density=300)
                        naming(test=True, folder_name=foldername, remove_SAGE_TMP=remove_SAGE_TMP) 
                    elif label != None:
                        self.quiver(tikz=tikz, scale=scale, label=label).png(filename, density=300)
                        naming(test=True, folder_name=foldername, remove_SAGE_TMP=remove_SAGE_TMP)
                    
                    print("Saved the file at "+str(os.getcwd())+"/"+foldername)
                    print("")

                elif slabbe_check == False:
                    print("The package `slabbe` may not be installed correctly. This package needs `dot2tex`, `glucose` and `latte_int`.")
                    print("Run ./sage --pip3 install slabbe")

            return TikzPicture.from_graph_with_pos(G, scale=scale)


    def show_vertex(self):
        """
            Display the vertex set
        """
        I = self._cartan_type.index_set()
        W = self._vertex_set

        print("")
        print("Index".ljust(8)+" : { "+"Label(Numbering)".ljust(19)+" : "+"Vertex }")
        print("-------------------------------------------------")
        for i in I:
            for key, value in W[i].items():
                print("  "+str(i).ljust(15)+str(key).ljust(17)+str(value))
            print("")


    def show_current_mutations(self):
        """
            Display the current mutations
        """
        I = self._cartan_type.index_set()
        W = self._vertex_set

        print("")
        print("Label".ljust(11)+"Vertex".ljust(15)+"total count".ljust(20)+"order")
        print("---------------------------------------------------------")
        for idx in I:
            for key in W[idx].keys():
                for k, v in self._count_mutations[key].items():
                    if k[1] == 0:
                        print("  "+str(key).ljust(8)+" "+str(k).ljust(15)+str(v[0]).ljust(20)+str(v[1]))
                    else:
                        print("  "+str(key).ljust(8)+" "+str(k).ljust(15)+str(v[0]).ljust(20)+str(v[1]))
            print("")


    def show_current_cluster_variables(self):
        I = self._cartan_type.index_set()
        W = self._vertex_set
        cv = self._cluster
        length = []
        for var in cv:
            length.append(len(str(var)))
        max_length = max(length)

        print("")
        top = "Index".ljust(11)+"Label".ljust(15)+"Vertex".ljust(15)+"cluster variable"
        print(top)
        for idx in I:
            for key in W[idx].keys():
                expr = "  "+str(idx).ljust(10)+str(key).ljust(14)+"("+str(W[idx][key][0])+","+str(W[idx][key][1]).rjust(3)+")".ljust(9)+str(cv[key]).ljust(max_length)
                print(expr)
        print("")


    def mutation(self, i, show=False):
        """
            INPUT and DESCRIPTION:
                `i` : mutation at vertex with labeliing `i`

            DEPENDENCY:
                https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/cluster_algebra_quiver/quiver_mutation_type.html
        """
        A = self._cluster_algebra
        N = len(A.cluster())
        l = len(self._initial_matrix.columns())

        # Error
        if i not in range(N):
            print("Out of ragne:", end=' ')
            return 0

        else:
            if show == True:
                print(self._matrix)
                print("")
                print(self._cluster)
                print("")
                print("|".rjust(2*l-round(l/2)))
                print(" ".rjust(round(l/2))+"mutate at "+str(i))
                print("|".rjust(2*l-round(l/2)))
                print("")
            
            A.mutate(i)

            # ! Update the exchange matrix, clusters, and mutation count
            self._matrix = A.b_matrix()
            self._cluster = A.cluster()

            if show == True:
                print(self._matrix)
                print("")
                print(self._cluster)

            for label in self._count_mutations.keys():
                if label == i:
                    for vertex in self._count_mutations[label].keys():
                        self._count_mutations[label][vertex][0] += 1
                        self._order_mutation += 1
                        self._count_mutations[label][vertex][1].append(self._order_mutation)


    def mutations(self, seq, save=False, progress=False, scale=2, remove_SAGE_TMP=False):
        """"
            INPUT:
                `seq` : a sequence of mutations
                `save` (optional, default = False) : save the mutations of tikz type 
                (long time)
                'progress' (optional, default = False) : display the current progess 
                                                         of mutations by using opt  ional package `tqdm`
                `scale` (optional, default = 2) : scale for a picture
                `remove_SAGE_TMP` (optional, default = False) : initial the folder in `SAGE_TMP`

            DESCRIPTION:
                Mutate along `seq`.
                If there is a value `k` in `seq`, which is not contained in the indices of vertex of `G-`, then it is ignored.
        """
        ct = self._cartan_type
        W = self._vertex_set
        I = ct.index_set()
        label = []
        count = 0
        error = 0

        for idx in I:
            for key in self._vertex_set[idx]:
                label.append(key)

        if progress == True:
            if tqdm_check == True:
                seq = tqdm(seq)
            elif tqdm_check == False:
                pass 
            
        if save  == True:
            if slabbe_check == True:
                filename0 = tmp_filename('mutation_'+str(0)+'_'+'initial'+'_','.png')
                self.quiver(tikz=True, scale=scale).png(filename0, density=300)
        else:
            error += 1
        
        for k in seq:
            if k in label:
                if progress == True and tqdm_check == True:
                    seq.set_description("Mutation at %s th vertex" % k) 
                self.mutation(k)
                count += 1
                
                if save == True:
                    if slabbe_check == True:
                        for idx in I:
                            for key in self._vertex_set[idx].keys():
                                if k == key:
                                    filename = tmp_filename('mutation_'+str(count)+'_'+str(self._vertex_set[idx][key])+'_','.png')
                                    self.quiver(tikz=True, scale=scale, label=k).png(filename, density=300)
                    
                    else:
                        error += 1
        
        if save == True and error == 0:
            foldername = "Result"+"_"+str(self._cartan_type[0])+str(self._cartan_type[1])+"_"+str(self._max)
            print("Finished to apply the mutations.", end=' ')
            naming(test=True, folder_name=foldername, remove_SAGE_TMP=remove_SAGE_TMP)
            print("Saved the files at "+str(os.getcwd())+"/"+str(foldername)+".")
            print("")

        elif save == True and error != 0:
            print("The mutations are applied to the given quiver, but since the optional package `slabbe` may not be installed, we couldn't save them.")
            print("Run ./sage --pip3 install slabbe")
            print("(This package needs the optional packages `dot2tex`, `glucose` and `latte_int`.)")
            print("")


    def mu(self, MAX):
        """
            INPUT:
                `MAX`  : maximal length of sequence of vertex

            DESCRIPTION:
                Return the mutation sequence `mu` of `self` (following the notation in [HL16])

            The number of columns in `mu`:
                Type A_n, D_n and E_n (n=6,7,8) : n
                Type B_n, C_n and F_n (n=4)     : 2*n
                Type G_2                          3*2 (=6)
        """
        ct = self._cartan_type
        n = ct.rank()
        I = ct.index_set()
        J = []
        t = self._t
        max_col = t*n # a sequence of t*n columns
        sequence = {}
        vertex = {}
        reverse_W = {}
        mutation_sequence_mu = []
        V = W_minus(ct, MAX)
        W = W_minus(ct, MAX)

        for idx in I:
            J.append(idx)

        # ! In a non-symmetric type, for an index involved multiple edge in the Dynkin diagram, 
        # ! we separate the corresponding vertex set following the convention of [HL, 2016]:
        if ct[0] == 'C':
            W[n+1] = {}
            for key, value in V[n].items():
                if value[1] % 4 == 2:
                    W[n+1].update({key:value})
                    del(W[n][key])
            J.append(n+1)

        elif ct[0] == 'B':
            for idx in I:
                if idx != n:
                    W[-idx] = {}
            
            for idx in I:
                if idx != n:
                    for key, value in V[idx].items():
                        if value[1] % 4 == 3:
                            W[-idx].update({key:value})
                            del(W[idx][key])
                    J.append(-idx)

        elif ct[0] == 'F':
            for idx in I:
                if idx == 3 or idx == 4:
                    W[-idx] = {}
            
            for idx in I:
                if idx not in (1,2):
                    for key, value in V[idx].items():
                        if value[1] % 4 == 2:
                            W[-idx].update({key:value})
                            del(W[idx][key])
                    J.append(-idx)

        # initialize
        for col in range(max_col):
            sequence[col+1] = {}

        for idx in J:
            vertex[idx] = []

        for idx in J:
            for V in W[idx].values():
                vertex[idx].append(V)

        for idx in J:
            for value in W[idx].values():
                reverse_W[value] = idx

        # Due to technical reason, we consider type G separately.
        if ct[0] != 'G':
            for col in range(max_col):
                compare = []
                for idx in J:
                    compare.append(vertex[idx][0])

                compare_parameter = []
                for v in compare:
                    compare_parameter.append(v[1])
            
                for v in compare:
                    if v[1] == max(compare_parameter):
                        sequence[col+1] = W[reverse_W[v]]
                        for idx in I:
                            if ct[0] in ['A','D','E']:
                                if v[0] == idx:
                                    vertex[idx].remove(v)

                            if ct[0] == 'C':
                                if v[0] == idx and idx != n:
                                    vertex[idx].remove(v)
                                elif v[0] == idx and idx == n:
                                    if v[1] % 4 == 0:
                                        vertex[idx].remove(v)
                                    elif v[1] % 4 == 2:
                                        vertex[idx+1].remove(v)

                            if ct[0] == 'B':
                                if v[0] == idx and idx == n:
                                    vertex[idx].remove(v)
                                elif v[0] == idx and idx != n:
                                    if v[1] % 4 == 1:
                                        vertex[idx].remove(v)
                                    elif v[1] % 4 == 3:
                                        vertex[-idx].remove(v)

                            if ct[0] == 'F':
                                if v[0] == idx and idx in [1,2]:
                                    vertex[idx].remove(v)
                                elif v[0] == idx and idx not in [1,2]:
                                    if v[1] % 4 == 0:
                                        vertex[idx].remove(v)
                                    elif v[1] % 4 == 2:
                                        vertex[-idx].remove(v)
                        break
        
        if ct[0] == 'G':
            standard = [2,1,2,1,2,1]
            W['a'] = {}
            W['b'] = {}
            W['c'] = {}
            for label, vertex in W[1].items():
                if vertex[1] % 6 == 5:
                    W['a'].update({label:vertex})
                elif vertex[1] % 6 == 3:
                    W['b'].update({label:vertex})
                elif vertex[1] % 6 == 1:
                    W['c'].update({label:vertex})
            del(W[1])

            for col in range(max_col):
                if standard[col] == 2:
                    sequence[col+1] = W[2]
                if standard[col] == 1:
                    if col == 1:
                        sequence[col+1] = W['a']
                    elif col == 3:
                        sequence[col+1] = W['b']
                    elif col == 5:
                        sequence[col+1] = W['c']

        for col in range(max_col):
            for label in sequence[col+1]:
                mutation_sequence_mu.append(label)

        return mutation_sequence_mu


    def generator_monomials(self, show=False):
        """ 
            INPUT:
                `ct` : a Cartan type
                `max_num` : maximal length of monomial

            DESCRIPTION:
                Return the monomials of Y's attached to `initial` cluster variables.
                Note
                    `z_{i,r} <---(1-1)---->   prod_{k >= 0, r+kb_{ii} <= 0} Y_{i, r+b_ii}
        """
        ct = self._cartan_type
        I = CartanType(ct).index_set()
        B = self._B
        W = self._vertex_set
        rW = {}
        for idx in I:
            rW[idx] = {v:k for k,v in W[idx].items()}
        spectral_parameters = {}
        mono_list = deepcopy(W)

        for idx in I:
            spectral_parameters[idx] = []

        for idx in I:
            for value in W[idx].values():
                spectral_parameters[idx].append(value)

        for idx in I:
            for sp in spectral_parameters[idx]:
                x = Monomial({(sp[0], sp[1]):1})
                for k in numpy.arange(1, -sp[1] / B[idx-1][idx-1] + 0.001):
                    x *= Monomial({(sp[0], sp[1]+int(k)*B[idx-1][idx-1]):1})
                
                mono_list[idx][rW[idx].get((sp[0], sp[1]))] = x

        if show == True:
            for idx in I:
                for key, value in mono_list.items():
                    print(key, ":", value)
                    

        return mono_list


    def show_initial_monomials(self):
        I = self._cartan_type.index_set()
        W = self._vertex_set

        print("")
        for idx in I:
            for key in W[idx].keys():
                if W[idx][key][1] > 0 or W[idx][key][1] == 0:
                    print("y_{"+str(W[idx][key][0])+", "+str(W[idx][key][1])+"}^{(0)} = ".ljust(3), self.generator_monomials()[idx][key])
                elif W[idx][key][1] < 0:
                    print("y_{"+str(W[idx][key][0])+","+str(W[idx][key][1])+"}^{(0)} = ".ljust(3), self.generator_monomials()[idx][key])
            print("")


    def change_of_variables(self, i):
        """
        INPUT:
            `i` : a `i`-th cluster variable

        DESCRIPTION:
            Return a Laurent polynomial in terms of variable `Y_{i,r}` obtained from    `laurent` by change of variables
        """
        ct = self._cartan_type
        I = ct.index_set()
        gen = self.generator_monomials()
        gen_list = {}
        m1 = self._cluster[i].numerator()
        m2 = self._cluster[i].denominator()
        mono_list = []

        for idx in I:
            for key, value in gen[idx].items():
                gen_list[key] = value
        
        for value in dict(sorted(gen_list.items())).values():
            mono_list.append(value)

        numerator = convert(ct, m1, mono_list)
        denominator = convert(ct, m2, mono_list)

        return (numerator * inverse_of_monomial(ct, denominator))


# * Technical parts
def convert(ct, m, mono_list):
    """
    FUNCTION:
        Substitute the cluster variables to variable Y's
    
    INPUT:
        ct : Cartan type
        m :  a polynomial with multi-variables obtained from quiver mutations
        max_num : the number of cluster variables
        mono_list : the list of monomials of X
    """
    N = qCharacter({Monomial({}):1},ct)
    for s in range(0,len(list(m))):
        n = qCharacter({Monomial({}):1},ct)
        p = list(m)[s][1]

        M = []
        for k in range(len(list(p.dict().keys())[0])):
        # if k <= max_num:
            power = list(p.dict().keys())[0][k]
            if power != 0:
                M.append(qCharacter({mono_list[k]:1},ct)**power)

        for monomial in M:
            n *= monomial
        
        if s == 0:
            N = n*list(m)[s][0]
        else:
            N += n*list(m)[s][0]
    return N


def inverse_of_monomial(ct, m):
    """    
    INPUT:
        `ct` : Cartan type
        `m` : a `q`-character consisting of a single monomial
        
    DESCRIPTION:
        Return `m^-1`
    """
    DI = {}
    for monomial in m._poly_dict:
        for key, value in monomial._data.items():
            DI[key] = -value
            
    return qCharacter({Monomial(DI):1}, ct)


# * for compare exchange matrices
def compare(M, N, show=False):
    """
        INPUT:
            `M`, `N`: matrix
            `show` (optional, default : False) : show the different entries

        DESCRIPTION:
            Compare two matrix `M` and `N`
    """
    M_rows = len(M.rows())
    M_columns = len(M.columns())
    N_rows = len(N.rows())
    N_columns = len(N.columns())
    diff = {}
    count = 0

    if M_rows != N_rows or M_columns != N_columns:
        print("They have different size:", end=' ')
        return False

    else:
        for i in range(M_rows):
            for k in range(M_columns):
                if M[i, k] != N[i, k]:
                    diff[(i,k)] = [M[i,k], N[i,k]]
                    count += 1

        if show==True:
            if count > 0:
                print("before".ljust(11)+"after".ljust(12)+"initial matrix".ljust(20)+"mutated matrix")
                for key, value in diff.items():
                    if value[0] > 0:
                        print(str(key[0])+"->"+str(key[1]).ljust(9), end='')
                    elif value[0] < 0:
                        print(str(key[0])+"<-"+str(key[1]).ljust(9), end='')
                    elif value[0] == 0:
                        print(str(key[0])+"  "+str(key[1]).ljust(9), end='')

                    if value[1] > 0:
                        print(str(key[0])+"->"+str(key[1]).ljust(9), end='')
                    elif value[1] < 0:
                        print(str(key[0])+"<-"+str(key[1]).ljust(9), end='')
                    elif value[1] == 0:
                        print(str(key[0])+"  "+str(key[1]).ljust(9), end='')

                    print(str(value[0]).ljust(20)+str(value[1]))

            elif count == 0:
                print("They are same:",end=' ')
                return True

        elif show==False:
            if count > 0:
                print("They are different:",end='\n')
                return diff
            if count == 0:
                print("They are same:",end=' ')
                return True



class Monomial(SageObject):
    """
    A monomial of a `q`-character.
    """
    def __init__(self, d):
        self._data = dict(d)


    def __hash__(self):
        return hash(tuple(sorted(self._data.items())))


    def __eq__(self, other):
        return type(self) == type(other) and self._data == other._data


    def __ne__(self, other):
        return not (self == other)


    def __mul__(self, other):
        if len(self._data) < len(other._data):
            self, other = other, self

        ret = Monomial(self._data)
        for k in other._data:
            if k in ret._data:
                ret._data[k] += other._data[k]
                if ret._data[k] == 0:
                    del ret._data[k]
            else:
                ret._data[k] = other._data[k]
        return ret


    def __truediv__(self, other):
        if other == 0:
            return False
        else:
            ret = Monomial(self._data)

            for k in other._data:
                if k in ret._data:
                    ret._data[k] -= other._data[k]
                    if ret._data[k] == 0:
                        del ret._data[k]
                else:
                    ret._data[k] -= other._data[k]
        return ret


    def _repr_(self):
        if not self._data:
            return '1'
        def exp_repr(e):
            if e == 1:
                return ''
            return '^%s' % e
        return '*'.join('Y{}[{}]'.format(i, qpow) + exp_repr(self._data[i,qpow])
                        for i,qpow in sorted(self._data))


    def _latex_(self):
        if not self._data:
            return '1'
        def exp_repr(e):
            if e == 1:
                return ''
            return '^{%s}' % e
        def q_exp(e):
            if e == 0:
                return '1'
            if e == 1:
                return 'q'
            return 'q^{{{}}}'.format(e)
        return ' '.join('X_{{{},{}}}'.format(i, q_exp(qpow)) + exp_repr(self._data[i,qpow])
                        for i,qpow in sorted(self._data))


    def Y_iq_powers(self, i):
        """
        Return a ``dict`` whose keys are the powers of `q` for the variables
        `Y_{i,q^k}` occuring in ``self`` and the values are the exponents.
        """
        return {k[1]: self._data[k] for k in self._data if k[0] == i}


    def mult_Y_iq(self, i, k, e):
        """
        Multiply (and mutate) ``self`` by `Y_{i,q^k}^e`.
        """
        if (i, k) in self._data:
            if self._data[i,k] == -e:
                del self._data[i,k]
            else:
                self._data[i,k] += e
        else:
            self._data[i,k] = e


    def is_dominant(self):
        return all(k >= 0 for k in self._data.values())


    def is_antidominant(self):
        return all(k <= 0 for k in self._data.values())


class qCharacter(SageObject):
    """
    A `q`-character.
    """
    def __init__(self, d, cartan_type):
        self._poly_dict = dict(d)
        self._cartan_type = cartan_type


    def _repr_(self):
        if not self._poly_dict:
            return "0"
        def coeff(c):
            if c == 1:
                return ''
            return repr(c) + '*'
        return " + ".join(coeff(self._poly_dict[m]) + repr(m) if m._data
                          else repr(self._poly_dict[m])
                          for m in self._poly_dict)


    def _latex_(self):
        if not self._poly_dict:
            return "0"
        def coeff(c):
            if c == 1:
                return ''
            return latex(c)
        return " + ".join(coeff(self._poly_dict[m]) + latex(m) if m._data
                          else latex(self._poly_dict[m])
                          for m in self._poly_dict)


    def __eq__(self, other):
        return self._poly_dict == other._poly_dict


    def __len__(self):
        """
        Return the number of monomials of ``self``.
        """
        return sum(self._poly_dict.values())


    def __getitem__(self, m):
        if m not in self._poly_dict:
            return 0
        else:
            return self._poly_dict[m]


    def __iter__(self):
        """
        Iterate over the monomials of ``self``.
        """
        return iter(self._poly_dict)


    def __add__(self, other):
        assert other._cartan_type is self._cartan_type
        return qCharacter(blas_dict_add(self._poly_dict, other._poly_dict), self._cartan_type)


    def __sub__(self, other):
        return qCharacter(axpy(-1, self._poly_dict, other._poly_dict), self._cartan_type)


    def __mul__(self, other):
        if not isinstance(other, qCharacter):
            # Assume it is a scalar
            d = self._poly_dict
            return qCharacter({m: other * d[m] for m in d}, self._cartan_type)

        ret = qCharacter({}, self._cartan_type)
        for m in self._poly_dict:
            for mp in other._poly_dict:
                mpp = m * mp
                if mpp in ret._poly_dict:
                    ret._poly_dict[mpp] += self._poly_dict[m] * other._poly_dict[mp]
                else:
                    ret._poly_dict[mpp] = self._poly_dict[m] * other._poly_dict[mp]
        return ret


    def __pow__(self, other):
        M = self
        for k in range(0,other-1):
            M *= self
        
        return M


    def __truediv__(self, other):
        if other == 0:
            return False
        
        if len(other._poly_dict) == 1:
            ret = qCharacter({}, self._cartan_type)
            for m in self._poly_dict:
                for mp in other._poly_dict:
                    mpp = m / mp
                    if mpp in ret._poly_dict:
                        ret._poly_dict[mpp] += self._poly_dict[m] / other._poly_dict[mp]
                    else:
                        ret._poly_dict[mpp] = self._poly_dict[m] / other._poly_dict[mp]
                return ret
        else:
            return False


    @cached_method
    def dominant_monomials(self):
        return tuple([m for m in self._poly_dict if m.is_dominant()])


    def antidominant_monomials(self):
        return tuple([m for m in self._poly_dict if m.is_antidominant()])


    def is_le(self, m, mp):
        """
        Return if ``m`` is less than ``mp``.
        """
        Q = RootSystem(self._cartan_type).root_space()
        La = Q.fundamental_weights_from_simple_roots()
        wt = Q.sum(La[i] for i,k in m._data) - Q.sum(La[i] for i,k in mp._data)
        return all(wt[i] >= 0 for i in self._cartan_type.index_set())


# * file handling
def naming(test=False, folder_name=None, remove_SAGE_TMP=False):
    """
        INPUT:
            - test (optional, default= False) : for test
            - forder_name (optional, default= None) : to name the folder in which results are saved.
            - remove_SAGE_TMP (optional, default= False) : remove the temporary files

        DESCRIPTION:
            To rename the resulting files and move them in the folder located in the source file.
    """
    file_list = os.listdir(SAGE_TMP)
    image_list = [i for i in file_list if ('.png' in i)]
    tex_list = [j for j in file_list if ('.tex' in j)] 
    rename_list = {}
    
    for name in image_list:
        rename = name.split('_')
        rename_list[name] = rename

    margin = 10
    font = ImageFont.truetype(font="Courier", size=int(30))
    
    if test == True:
        if folder_name == None:
            if os.path.isdir(f"{os.getcwd()}"+"/Results") == False:
                os.mkdir(os.getcwd()+"/"+"Results")
        
        elif folder_name != None:
            if os.path.isdir(f"{os.getcwd()}"+"/"+folder_name) == False:
                os.mkdir(os.getcwd()+"/"+folder_name)
        
        if tqdm_check == True:
            image_list = tqdm(image_list)
            image_list.set_description("Processing for naming") 
        elif tqdm_check == False:
            pass

        for file in image_list:
            if 'quiver' in f"{rename_list[file][0]}" or 'mutation' in f"{rename_list[file][0]}":
                img = Image.open(f"{SAGE_TMP}/{file}") 
                width, height = img.size
                background = Image.new("RGB", (width+50, height+100), (255,255,255))
                background.paste(img, (25, 60))
                background.save(f"{SAGE_TMP}/{file}")
            
                img = Image.open(f"{SAGE_TMP}/{file}")
                draw = ImageDraw.Draw(img)
                text = f"{rename_list[file][0]}"+" at "+f"{rename_list[file][2]}"+":"+f"{rename_list[file][1]}" 
                width, height = img.size
                width_txt, height_txt = draw.textsize(text, font)
            
                x = width - width_txt - margin
                y = height_txt - margin
            
                if f"{rename_list[file][2]}" == "initial":
                    draw.text((x,y), "Initial quiver", font=font, fill="#000")
                elif 'quiver' in f"{rename_list[file][0]}":
                    pass
                else: 
                    draw.text((x,y), text, font=font, fill="#000")
            
                draw = ImageDraw.Draw(img)
            
                if folder_name == None: 
                    img.save(os.getcwd()+"/Results/"+f"{file.split('_')[0]+file.split('_')[1]+file.split('_')[2]}.png")
                elif folder_name != None:
                    img.save(os.getcwd()+"/"+folder_name+"/"+f"{file.split('_')[0]+file.split('_')[1]+file.split('_')[2]}.png")

        for tex in tex_list:
            if folder_name == None:
                shutil.move(f"{SAGE_TMP}/{tex}", os.getcwd()+"/Results/")
            elif folder_name != None:
                shutil.move(f"{SAGE_TMP}/{tex}", os.getcwd()+"/"+folder_name+"/")
        
        if remove_SAGE_TMP == True:
            shutil.rmtree(f"{SAGE_TMP}")