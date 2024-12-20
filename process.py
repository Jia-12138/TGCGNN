
#####################  下面这里就是属于
import json
import os
import sys
import csv
import numpy as np
import torch

from ase import io
from ase.db import connect
from pymatgen.core.structure import Structure
from scipy.stats import rankdata
# from pymatgen.io import ase
from torch_geometric.utils import dense_to_sparse, add_self_loops
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch.nn.functional as F
from torch_geometric.utils import degree
import ase
import os
import sys
import time
import csv
import json
import warnings
import numpy as np
import ase
import glob
from ase import io
from scipy.stats import rankdata
from scipy import interpolate

##torch imports
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree
def split_data(
        dataset,
        train_ratio,
        val_ratio,
        test_ratio,
        seed=np.random.randint(1, 1e6),
        save=False,
):
    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <= 1:
        train_length = int(dataset_size * train_ratio)
        val_length = int(dataset_size * val_ratio)
        test_length = int(dataset_size * test_ratio)
        unused_length = dataset_size - train_length - val_length - test_length
        (
            train_dataset,
            val_dataset,
            test_dataset,
            unused_dataset,
        ) = torch.utils.data.random_split(
            dataset,
            [train_length, val_length, test_length, unused_length],
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            "train length:",
            train_length,
            "val length:",
            val_length,
            "test length:",
            test_length,
            "unused length:",
            unused_length,
            "seed :",
            seed,
        )
        return train_dataset, val_dataset, test_dataset
    else:
        print("invalid ratios")
##Basic CV split
def split_data_CV(dataset, num_folds=5, seed=np.random.randint(1, 1e6), save=False):
    dataset_size = len(dataset)
    fold_length = int(dataset_size / num_folds)
    unused_length = dataset_size - fold_length * num_folds
    folds = [fold_length for i in range(num_folds)]
    folds.append(unused_length)
    cv_dataset = torch.utils.data.random_split(
        dataset, folds, generator=torch.Generator().manual_seed(seed)
    )
    print("fold length :", fold_length, "unused length:", unused_length, "seed", seed)
    return cv_dataset[0:num_folds]

def get_dataset(data_path, target_index, reprocess="False", processing_args=None):
    if processing_args == None:
        processed_path = "processed"
    else:
        processed_path = processing_args.get("processed_path", "processed")

    transforms = GetY(index=target_index)

    if os.path.exists(data_path) == False:
        print("Data not found in:", data_path)
        sys.exit()

    if reprocess == "True":
        os.system("rm -rf " + os.path.join(data_path, processed_path))
        process_data(data_path, processed_path, processing_args)

    if os.path.exists(os.path.join(data_path, processed_path, "data.pt")) == True:
        dataset = StructureDataset(
            data_path,
            processed_path,
            transforms,
        )
    elif os.path.exists(os.path.join(data_path, processed_path, "data0.pt")) == True:
        dataset = StructureDataset_large(
            data_path,
            processed_path,
            transforms,
        )
    else:
        process_data(data_path, processed_path, processing_args)
        if os.path.exists(os.path.join(data_path, processed_path, "data.pt")) == True:
            dataset = StructureDataset(
                data_path,
                processed_path,
                transforms,
            )
        elif os.path.exists(os.path.join(data_path, processed_path, "data0.pt")) == True:
            dataset = StructureDataset_large(
                data_path,
                processed_path,
                transforms,
            )
    return dataset

class StructureDataset(InMemoryDataset):
    def __init__(
            self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = ["data.pt"]
        return file_names
class StructureDataset_large(Dataset):
    def __init__(
            self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset_large, self).__init__(
            data_path, transform, pre_transform
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        # file_names = ["data.pt"]
        file_names = []
        for file_name in glob.glob(self.processed_dir + "/data*.pt"):
            file_names.append(os.path.basename(file_name))
        # print(file_names)
        return file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, "data_{}.pt".format(idx)))
        return data
################################################################################
#  Processing
################################################################################
def create_global_feat(atoms_index_arr):
    comp = np.zeros(108)
    temp = np.unique(atoms_index_arr, return_counts=True)
    for i in range(len(temp[0])):
        comp[temp[0][i]] = temp[1][i] / temp[1].sum()
    return comp.reshape(1, -1)
################################################################################
# 计算键角余弦值并将其封装到数据对象中
################################################################################



# import torch.nn as nn
# class RBFExpansion(nn.Module):
#     """Expand interatomic distances with radial basis functions."""
#
#     def __init__(self, vmin=-1, vmax=1, bins=40, lengthscale=None):
#         super().__init__()
#         self.vmin = vmin
#         self.vmax = vmax
#         self.bins = bins
#         self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))
#
#         if lengthscale is None:
#             self.lengthscale = torch.mean(self.centers[1:] - self.centers[:-1])
#             self.gamma = 1 / (self.lengthscale ** 2)
#         else:
#             self.lengthscale = lengthscale
#             self.gamma = 1 / (lengthscale ** 2)
#
#     def forward(self, distance):
#         return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)



import torch

def get_Kpoints_random(q,lattice,volume):
    a0=lattice[0,:]
    a1=lattice[1,:]
    a2=lattice[2,:]        ## 将晶胞3个行向量分别提取出来
    unit=2*np.pi*np.vstack((np.cross(a1,a2),np.cross(a2,a0),np.cross(a0,a1)))/volume
    ur=[(2*r-q-1)/2/q for r in range(1,q+1)]
    points=[]
    for i in ur:
        for j in ur:
            for k in ur:
                points.append(unit[0,:]*i+unit[1,:]*j+unit[2,:]*k)
    points=np.array(points)
    return points


def calculate_plane_wave(vectorij, K_points, volume):
    # 计算平面波函数展开
    kr = np.dot(vectorij, K_points.T)
    plane_wave = np.cos(kr) / np.sqrt(volume)
    return plane_wave
def calculate_vectorij(positions, edge_index):
    source_nodes = edge_index[0].numpy()    ## 源节点
    target_nodes = edge_index[1].numpy()    ## 目标节点
    vectorij = positions[source_nodes] - positions[target_nodes]
    return vectorij            ## 计算向量

def compute_edge_distances(edge_index,pos):
    # 获取所有边的节点对
    edge_sources = edge_index[0]  # 边的源节点
    edge_targets = edge_index[1]  # 边的目标节点

    # 获取节点的坐标 (num_nodes, 3)，3 是 x, y, z 坐标
    positions = pos

    # 通过源节点和目标节点的坐标计算边的向量
    vector_ij = positions[edge_sources] - positions[edge_targets]  # (num_edges, 3)

    # 计算每条边的欧几里得距离
    edge_distances = torch.norm(vector_ij, dim=1)  # (num_edges,)



    return edge_distances

## 定义四种用于产生combine_set的函数
def Phi(r,cutoff):
    return 1-6*(r/cutoff)**5+15*(r/cutoff)**4-10*(r/cutoff)**3
def gaussian(r,miuk,betak):
    return np.exp(-betak*(np.exp(-r)-miuk)**2)
def miuk(n,K,cutoff):
    # n=[1,K]
    return np.exp(-cutoff)+(1-np.exp(-cutoff))/K*n
def betak(K,cutoff):
    return (2/K*(1-np.exp(-cutoff)))**(-2)
###################

def process_data(data_path, processed_path, processing_args):
    ##Begin processing data
    print("Processing data to: " + os.path.join(data_path, processed_path))
    assert os.path.exists(data_path), "Data path not found in " + data_path

    ##Load dictionary
    if processing_args["dictionary_source"] != "generated":
        if processing_args["dictionary_source"] == "default":
            print("Using default dictionary.")
            atom_dictionary = get_dictionary(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "dictionary_default.json",
                )
            )
        elif processing_args["dictionary_source"] == "blank":
            print(
                "Using blank dictionary. Warning: only do this if you know what you are doing"
            )
            atom_dictionary = get_dictionary(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "dictionary_blank.json"
                )
            )
        else:
            dictionary_file_path = os.path.join(
                data_path, processing_args["dictionary_path"]
            )
            if os.path.exists(dictionary_file_path) == False:
                print("Atom dictionary not found, exiting program...")
                sys.exit()
            else:
                print("Loading atom dictionary from file.")
                atom_dictionary = get_dictionary(dictionary_file_path)

    ##Load targets
    target_property_file = os.path.join(data_path, processing_args["target_path"])
    assert os.path.exists(target_property_file), (
            "targets not found in " + target_property_file
    )
    with open(target_property_file) as f:
        reader = csv.reader(f)
        target_data = [row for row in reader]

    ##Read db file if specified
    ase_crystal_list = []
    if processing_args["data_format"] == "db":
        db = ase.db.connect(os.path.join(data_path, "data.db"))
        row_count = 0
        # target_data=[]
        for row in db.select():
            # target_data.append([str(row_count), row.get('target')])
            ase_temp = row.toatoms()
            ase_crystal_list.append(ase_temp)
            row_count = row_count + 1
            if row_count % 500 == 0:
                print("db processed: ", row_count)

    ##Process structure files and create structure graphs
    data_list = []
    for index in range(0, len(target_data)):

        structure_id = target_data[index][0]
        data = Data()

        ##Read in structure file using ase
        if processing_args["data_format"] != "db":
            ase_crystal = ase.io.read(
                os.path.join(
                    data_path, structure_id + "." + processing_args["data_format"]

                )
            )
            data.ase = ase_crystal
        else:
            ase_crystal = ase_crystal_list[index]
            data.ase = ase_crystal

        ## Step 1: 获取晶格和体积信息
        lattice = ase_crystal.cell.array  # 晶格矩阵   ## (3,3)
        volume = ase_crystal.get_volume()  # 晶体体积
        positions = ase_crystal.get_positions()  # 原子坐标 ##(10,3)
        ## 波函数，需要一个高斯，需要晶胞参数和晶体体积，还有一个向量（边的个数，3）
        grid = n_grid_K=4
        # kr = np.dot(self.vectorij, get_Kpoints_random(grid, lattice, volume).transpose())


        # ## Step 4: 生成原子对的位移向量 (Vectorij)
        # distance_matrix = ase_crystal.get_all_distances(mic=True)   ## （10，10）
        # distance_matrix_trimmed = threshold_sort(
        #     distance_matrix,
        #     processing_args["graph_max_radius"],
        #     processing_args["graph_max_neighbors"],
        #     adj=False,
        # )
        # distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        # out = dense_to_sparse(distance_matrix_trimmed)
        # edge_index = out[0]
        # vectorij = calculate_vectorij(positions, edge_index)  ## (边的个数，3)
        # kr = np.dot(vectorij, get_Kpoints_random(grid, lattice, volume).transpose())   ## 这里的grid就是q
        # plane_wave = np.cos(kr) / np.sqrt(volume)
        #
        # ## 将平面波信息封装到 data 对象中
        # data.plane_wave = torch.Tensor(plane_wave)

        ##Compile structure sizes (# of atoms) and elemental compositions
        if index == 0:
            length = [len(ase_crystal)]
            elements = [list(set(ase_crystal.get_chemical_symbols()))]
        else:
            length.append(len(ase_crystal))
            elements.append(list(set(ase_crystal.get_chemical_symbols())))

        ##获取距离矩阵
        distance_matrix = ase_crystal.get_all_distances(mic=True)




        ##使用距离矩阵创建稀疏图
        distance_matrix_trimmed = threshold_sort(
            distance_matrix,
            processing_args["graph_max_radius"],
            processing_args["graph_max_neighbors"],
            adj=False,
        )

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]     ## 转换成连接
        # 提取与边对应的距离
        edge_distances = distance_matrix[edge_index[0], edge_index[1]]
        # 将距离信息封装到 self.distance 中
        # data.distance = torch.Tensor(edge_distances)      ## 72，等一会就让你滚出去
        edge_weight = out[1]    ## 每条连接边的权重

        ## 只修改了这个是否自环的参数
        self_loops = False
        if self_loops == True:
            ## 自环信息添加到最后了
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
            )
            data.edge_index = edge_index
            data.edge_weight = edge_weight

            distance_matrix_mask = (
                    distance_matrix_trimmed.fill_diagonal_(1) != 0
            ).int()
            # 将距离信息封装到 self.distance 中
            edge_distances = distance_matrix[edge_index[0], edge_index[1]]
            # data.distance = torch.Tensor(edge_distances)


        elif self_loops == False:
            data.edge_index = edge_index
            data.edge_weight = edge_weight

            distance_matrix_mask = (distance_matrix_trimmed != 0).int()

        data.edge_descriptor = {}
        data.edge_descriptor["distance"] = edge_weight
        data.edge_descriptor["mask"] = distance_matrix_mask

        target = target_data[index][1:]
        y = torch.Tensor(np.array([target], dtype=np.float32))
        data.y = y
        ## 下面这个就是来使用自环函数进行波函数的计算
        vectorij = calculate_vectorij(positions, edge_index)  ## (边的个数，3)
        kr = np.dot(vectorij, get_Kpoints_random(grid, lattice, volume).transpose())  ## 这里的grid就是q
        # 初始化 plane_wave，并直接设置为全 0 数组
        plane_wave = np.zeros_like(kr)

        # 找到 vectorij 不为 0 的位置
        non_zero_mask = np.any(vectorij != 0, axis=1)

        # 对于 vectorij 不为 0 的部分，计算平面波
        plane_wave[non_zero_mask] = np.cos(kr[non_zero_mask]) / np.sqrt(volume)

        # 将 plane_wave 封装到 data 对象中
        data.plane_wave = torch.Tensor(plane_wave)
        # plane_wave = np.cos(kr) / np.sqrt(volume)

        ## 将平面波信息封装到 data 对象中
        data.plane_wave = torch.Tensor(plane_wave)

        ## 下面这部分代码是用于生成combine_set的
        # combine_sets = []
        # # gaussian radial
        # N = n_Gaussian = 64  ## 64
        # ## 1-65一共就是64次
        # for n in range(1, N + 1):
        #     phi = Phi(edge_distances, cutoff=8)
        #     G = gaussian(edge_distances, miuk(n, N, cutoff=8), betak(N, cutoff=8))
        #     # print(f"phi shape: {phi.shape}, G shape: {G.shape}")
        #     combine_sets.append(phi * G)
        # data.combine_sets = np.array(combine_sets, dtype=np.float32).transpose()



        ## 下面两行是用于获取原子序数之后并且进行得到元素的含量
        _atoms_index = ase_crystal.get_atomic_numbers()   ## 获取原子的序数
        gatgnn_glob_feat = create_global_feat(_atoms_index)
        gatgnn_glob_feat = np.repeat(gatgnn_glob_feat, len(_atoms_index), axis=0)
        data.glob_feat = torch.Tensor(gatgnn_glob_feat).float()

        pos = torch.Tensor(ase_crystal.get_positions())
        data.pos = pos
        ## 以下是用来计算论文中的rij向量
        edge_distances_r = compute_edge_distances(data.edge_index, data.pos)
        data.edge_distances_r = torch.Tensor(edge_distances_r)


        z = torch.LongTensor(ase_crystal.get_atomic_numbers())
        data.z = z

        ###placeholder for state feature
        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        data.u = u

        data.structure_id = [[structure_id] * len(data.y)]




        if processing_args["verbose"] == "True" and (index + 1) % 500 == 0:
            print("Data processed: ", index + 1, "out of", len(target_data))

        data_list.append(data)

    ##
    n_atoms_max = max(length)
    species = list(set(sum(elements, [])))
    species.sort()

    num_species = len(species)
    if processing_args["verbose"] == "True":
        print(
            "Max structure size: ",
            n_atoms_max,
            "Max number of elements: ",
            num_species,
        )
        print("Unique species:", species)
    crystal_length = len(ase_crystal)
    data.length = torch.LongTensor([crystal_length])

    ##Generate node features
    if processing_args["dictionary_source"] != "generated":
        ##Atom features(node features) from atom dictionary file
        for index in range(0, len(data_list)):
            atom_fea = np.vstack(
                [
                    atom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
                    for i in range(len(data_list[index].ase))
                ]
            ).astype(float)
            data_list[index].x = torch.Tensor(atom_fea)
    elif processing_args["dictionary_source"] == "generated":
        ##Generates one-hot node features rather than using dict file
        from sklearn.preprocessing import LabelBinarizer

        lb = LabelBinarizer()
        lb.fit(species)
        for index in range(0, len(data_list)):
            data_list[index].x = torch.Tensor(
                lb.transform(data_list[index].ase.get_chemical_symbols())
            )

    ##Adds node degree to node features (appears to improve performance)
    ## 这里如果不适用，先来看看
    # for index in range(0, len(data_list)):
    #     data_list[index] = OneHotDegree(
    #         data_list[index], processing_args["graph_max_neighbors"] + 1
    #     )
    #     # 处理 combine_sets
    #     combine_sets = []
    #     N = 64
    #     for n in range(1, N + 1):
    #         phi = Phi(edge_distances, cutoff=8)
    #         G = gaussian(edge_distances, miuk(n, N, cutoff=8), betak(N, cutoff=8))
    #         combine_sets.append(phi * G)
    #
    #     combine_sets = np.array(combine_sets, dtype=np.float32).transpose()
    #
    #     # 将 combine_sets 转换为 PyTorch 张量并保存到 data 中
    # data.combine_sets = torch.Tensor(combine_sets)

    ##Get graphs based on voronoi connectivity; todo: also get voronoi features
    ##avoid use for the time being until a good approach is found
    processing_args["voronoi"] = "False"
    if processing_args["voronoi"] == "True":
        from pymatgen.core.structure import Structure
        from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
        from pymatgen.io.ase import AseAtomsAdaptor

        Converter = AseAtomsAdaptor()

        for index in range(0, len(data_list)):
            pymatgen_crystal = Converter.get_structure(data_list[index].ase)
            # double check if cutoff distance does anything
            Voronoi = VoronoiConnectivity(
                pymatgen_crystal, cutoff=processing_args["graph_max_radius"]
            )
            connections = Voronoi.max_connectivity

            distance_matrix_voronoi = threshold_sort(
                connections,
                9999,
                processing_args["graph_max_neighbors"],
                reverse=True,
                adj=False,
            )
            distance_matrix_voronoi = torch.Tensor(distance_matrix_voronoi)

            out = dense_to_sparse(distance_matrix_voronoi)
            edge_index_voronoi = out[0]
            edge_weight_voronoi = out[1]

            edge_attr_voronoi = distance_gaussian(edge_weight_voronoi)
            edge_attr_voronoi = edge_attr_voronoi.float()

            data_list[index].edge_index_voronoi = edge_index_voronoi
            data_list[index].edge_weight_voronoi = edge_weight_voronoi
            data_list[index].edge_attr_voronoi = edge_attr_voronoi
            if index % 500 == 0:
                print("Voronoi data processed: ", index)

    ## 获取所有的描述符
    from dscribe.descriptors import SOAP, CoulombMatrix, SineMatrix, EwaldSumMatrix, ACSF, MBTR, LMBTR, ValleOganov

    if processing_args["SOAP_descriptor"] == "True":
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False

        from dscribe.descriptors import SOAP

        # rut=8
        make_feature_SOAP = SOAP(
            species=species,
            r_cut=8,
            n_max=processing_args["SOAP_nmax"],
            l_max=processing_args["SOAP_lmax"],
            sigma=processing_args["SOAP_sigma"],
            periodic=periodicity,
            sparse=False,
            average="off",
            # average="off",
            rbf="gto",
            compression={"mode": "mu2"}
        )
        for index in range(0, len(data_list)):
            features_SOAP = make_feature_SOAP.create(data_list[index].ase)
            data_list[index].extra_features_SOAP = torch.Tensor(features_SOAP)
            print(data_list[index].extra_features_SOAP.shape)
            if processing_args["verbose"] == "True" and index % 500 == 0:
                if index == 0:
                    print(
                        "SOAP length: ",
                        features_SOAP.shape,
                    )
                print("SOAP descriptor processed: ", index)

    elif processing_args["SM_descriptor"] == "True":
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False

        from dscribe.descriptors import SineMatrix, CoulombMatrix

        if periodicity == True:
            make_feature_SM = SineMatrix(
                n_atoms_max=n_atoms_max,
                permutation="eigenspectrum",
                sparse=False,
                flatten=True,
            )
        else:
            make_feature_SM = CoulombMatrix(
                n_atoms_max=n_atoms_max,
                permutation="eigenspectrum",
                sparse=False,
                flatten=True,
            )

        for index in range(0, len(data_list)):
            features_SM = make_feature_SM.create(data_list[index].ase)
            data_list[index].extra_features_SM = torch.Tensor(features_SM)
            if processing_args["verbose"] == "True" and index % 500 == 0:
                if index == 0:
                    print(
                        "SM length: ",
                        features_SM.shape,
                    )
                print("SM descriptor processed: ", index)

    ##Generate edge features
    if processing_args["edge_features"] == "True":

        ##Distance descriptor using a Gaussian basis
        distance_gaussian = GaussianSmearing(
            0, 1, processing_args["graph_edge_length"], 0.2
        )
        # print(GetRanges(data_list, 'distance'))
        NormalizeEdge(data_list, "distance")
        # print(GetRanges(data_list, 'distance'))
        for index in range(0, len(data_list)):
            data_list[index].edge_attr = distance_gaussian(
                data_list[index].edge_descriptor["distance"]
            )
            if processing_args["verbose"] == "True" and (
                    (index + 1) % 500 == 0 or (index + 1) == len(target_data)
            ):
                print("Edge processed: ", index + 1, "out of", len(target_data))

    Cleanup(data_list, ["ase", "edge_descriptor"])

    if os.path.isdir(os.path.join(data_path, processed_path)) == False:
        os.mkdir(os.path.join(data_path, processed_path))

    ##Save processed dataset to file
    if processing_args["dataset_type"] == "inmemory":
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), os.path.join(data_path, processed_path, "data.pt"))

    elif processing_args["dataset_type"] == "large":
        for i in range(0, len(data_list)):
            torch.save(
                data_list[i],
                os.path.join(
                    os.path.join(data_path, processed_path), "data_{}.pt".format(i)
                ),
            )


################################################################################
#  其他函数
################################################################################

##选择具有距离阈值和有限数量邻居的边
def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr



##基于高斯基的边描述符
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


##获取节点度数的独热编码表示
def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg

    return data


##从字典文件获取元素特征
def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


##删除不必要的数据以加快数据加载器
def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


##获取边的最小/最大范围以进行标准化
def GetRanges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max


##标准化边缘
def NormalizeEdge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
                                                         data.edge_descriptor[descriptor_label] - feature_min
                                                 ) / (feature_max - feature_min)


##生成 SM 描述符
def SM_Edge(dataset):
    from dscribe.descriptors import (
        CoulombMatrix,
        SOAP,
        MBTR,
        EwaldSumMatrix,
        SineMatrix,
    )

    count = 0
    for data in dataset:
        n_atoms_max = len(data.ase)
        make_feature_SM = SineMatrix(
            n_atoms_max=n_atoms_max,
            permutation="none",
            sparse=False,
            flatten=False,
        )
        features_SM = make_feature_SM.create(data.ase)
        features_SM_trimmed = np.where(data.mask == 0, data.mask, features_SM)
        features_SM_trimmed = torch.Tensor(features_SM_trimmed)
        out = dense_to_sparse(features_SM_trimmed)
        edge_index = out[0]
        edge_weight = out[1]
        data.edge_descriptor["SM"] = edge_weight

        if count % 500 == 0:
            print("SM data processed: ", count)
        count = count + 1

    return dataset


################################################################################
#  变换
################################################################################

##从 data.y 获取指定的 y 索引
class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data

