import h5torch
import numpy as np
import torch
import pytorch_lightning as pl
import warnings
from pytorch_lightning.callbacks import Callback


class dinucl_shuffled_negs(h5torch.Dataset):
    def __init__(
        self,
        file,
        TF,
        subset=None
    ):
        super().__init__(file)
        self.subset = subset
        self.TF = TF

        self.prot_mask = file["0/prot_names"][:] == TF.encode()
        self.central = file["central"][self.prot_mask].squeeze() #only take data for that TF
        self.pos_mask = self.central == 1 #only take the positive samples

        self.central = self.central[self.pos_mask] #subset data

        self.peak_ix_to_pos = file["1/peak_ix_to_pos"][self.pos_mask]
        self.peak_ix_to_len = file["1/peak_ix_to_len"][self.pos_mask]
        self.peak_ix_to_chr = file["1/peak_ix_to_chr"][self.pos_mask]

        
        pos_subset_mask = np.isin(self.peak_ix_to_chr.astype(str), subset) #only take the right (training, val) positions
        self.peak_ix_to_pos = self.peak_ix_to_pos[pos_subset_mask]
        self.peak_ix_to_len = self.peak_ix_to_len[pos_subset_mask]
        self.peak_ix_to_chr = self.peak_ix_to_chr[pos_subset_mask]

        self.mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
        self.rev_mapping = {v : k for k, v in self.mapping.items()}

        self.genome = {k : file["unstructured"][k] for k in list(file["unstructured"]) if k.startswith("chr")}
        self.file=file
        

        neg_subset_mask = np.isin(file["unstructured/dinucl_"+self.TF+"_chrs"][:].astype(str), subset) #only take the right (training, val) positions
        self.neg_seqs = file["unstructured/dinucl_"+self.TF+"_seqs"][neg_subset_mask] #subset the seqs
        self.neg_len = self.neg_seqs.shape[0] # number of negs



    def __len__(self):
        return len(self.peak_ix_to_pos)+self.neg_len # total number of datapoints is positive + negative samples
    

    def __getitem__(self, index):

        sample = {}
        sample["0/prot_names"] = self.TF
        sample["1/DNA_regions"] = []
        sample["central"] = []
        
        if index < len(self.peak_ix_to_pos): # take a positive
            chr = self.peak_ix_to_chr[index].decode()
            pos = self.peak_ix_to_pos[index]
            sample["1/DNA_regions"].append(self.genome[chr][pos-50:pos+51])
            sample["central"].append(1)

        else: # take a negative
        # get negative samples
            neg_seq = self.neg_seqs[index-len(self.peak_ix_to_pos)] #convert the index!
            sample["1/DNA_regions"].append(neg_seq)
            sample["central"].append(0)
        
        return sample
    

class dinucl_sampled_negs(h5torch.Dataset):
    def __init__(
        self,
        file,
        TF,
        subset=None
    ):
        #! still have to test this!
        super().__init__(file)
        self.subset = subset
        self.TF = TF

        self.prot_mask = file["0/prot_names"][:] == TF.encode()
        self.central = file["central"][self.prot_mask].squeeze() #only take data for that TF
        self.pos_mask = self.central == 1 #only take the positive samples

        self.central = self.central[self.pos_mask] #subset data

        self.peak_ix_to_pos = file["1/peak_ix_to_pos"][self.pos_mask]
        self.peak_ix_to_len = file["1/peak_ix_to_len"][self.pos_mask]
        self.peak_ix_to_chr = file["1/peak_ix_to_chr"][self.pos_mask]

        
        pos_subset_mask = np.isin(self.peak_ix_to_chr.astype(str), subset) #only take the right (training, val) positions
        self.peak_ix_to_pos = self.peak_ix_to_pos[pos_subset_mask]
        self.peak_ix_to_len = self.peak_ix_to_len[pos_subset_mask]
        self.peak_ix_to_chr = self.peak_ix_to_chr[pos_subset_mask]

        self.mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
        self.rev_mapping = {v : k for k, v in self.mapping.items()}

        self.genome = {k : file["unstructured"][k] for k in list(file["unstructured"]) if k.startswith("chr")}
        self.file=file
        

        neg_subset_mask = np.isin(self.file["unstructured/sampled_negs_"+TF+"_chr"][:].astype(str), self.subset) #only take the right (training, val) positions
        self.neg_chrs = file["unstructured/sampled_negs_"+TF+"_chr"][neg_subset_mask] #subset the chrs
        self.neg_pos = file["unstructured/sampled_negs_"+TF+"_pos"][neg_subset_mask] #subset the seqs
        self.neg_len = self.neg_pos.shape[0] # number of negs



    def __len__(self):
        return len(self.peak_ix_to_pos)+self.neg_len # total number of datapoints is positive + negative samples
    

    def __getitem__(self, index):

        sample = {}
        sample["0/prot_names"] = self.TF
        sample["1/DNA_regions"] = []
        sample["central"] = []
        
        if index < len(self.peak_ix_to_pos): # take a positive
            chr = self.peak_ix_to_chr[index].decode()
            pos = self.peak_ix_to_pos[index]
            sample["1/DNA_regions"].append(self.genome[chr][pos-50:pos+51])
            sample["central"].append(1)

        else: # take a negative
        # get negative samples
            converted_index = index-len(self.peak_ix_to_pos) #convert the index!
            chr = self.neg_chrs[converted_index].decode()
            pos = self.neg_pos[converted_index]
            sample["1/DNA_regions"].append(self.genome[chr][pos-50:pos+51])
            sample["central"].append(0)
        
        return sample

class shuffled_negs(h5torch.Dataset):
    def __init__(
        self,
        file,
        TF,
        subset=None
    ):
        #! still have to test this!
        super().__init__(file)
        self.subset = subset
        self.TF = TF

        self.prot_mask = file["0/prot_names"][:] == TF.encode()
        self.central = file["central"][self.prot_mask].squeeze() #only take data for that TF
        self.pos_mask = self.central == 1 #only take the positive samples

        self.central = self.central[self.pos_mask] #subset data

        self.peak_ix_to_pos = file["1/peak_ix_to_pos"][self.pos_mask]
        self.peak_ix_to_len = file["1/peak_ix_to_len"][self.pos_mask]
        self.peak_ix_to_chr = file["1/peak_ix_to_chr"][self.pos_mask]

        
        pos_subset_mask = np.isin(self.peak_ix_to_chr.astype(str), subset) #only take the right (training, val) positions
        self.peak_ix_to_pos = self.peak_ix_to_pos[pos_subset_mask]
        self.peak_ix_to_len = self.peak_ix_to_len[pos_subset_mask]
        self.peak_ix_to_chr = self.peak_ix_to_chr[pos_subset_mask]

        self.mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
        self.rev_mapping = {v : k for k, v in self.mapping.items()}

        self.genome = {k : file["unstructured"][k] for k in list(file["unstructured"]) if k.startswith("chr")}
        self.file=file


    def __len__(self):
        return len(self.peak_ix_to_pos)*2
    

    def __getitem__(self, index):

        sample = {}
        sample["0/prot_names"] = self.TF
        sample["1/DNA_regions"] = []
        sample["central"] = []
        
        if index < len(self.peak_ix_to_pos): # take a positive
            chr = self.peak_ix_to_chr[index].decode()
            pos = self.peak_ix_to_pos[index]
            sample["1/DNA_regions"].append(self.genome[chr][pos-50:pos+51])
            sample["central"].append(1)

        else: # take a negative
        # get negative samples
            converted_index = index-len(self.peak_ix_to_pos) #convert the index!
            chr = self.peak_ix_to_chr[converted_index]
            pos = self.peak_ix_to_pos[converted_index]
            DNA = self.genome[chr][pos-50:pos+51]
            shuffled_DNA =  np.random.permutation(DNA)
            sample["1/DNA_regions"].append(shuffled_DNA)
            sample["central"].append(0)
        
        return sample
  
class neighbor_negs(h5torch.Dataset):
    def __init__(
        self,
        file,
        TF,
        subset=None
    ):
        #! still have to test this!
        super().__init__(file)
        self.subset = subset
        self.TF = TF

        self.prot_mask = file["0/prot_names"][:] == TF.encode()
        self.central = file["central"][self.prot_mask].squeeze() #only take data for that TF
        self.pos_mask = self.central == 1 #only take the positive samples

        self.central = self.central[self.pos_mask] #subset data

        self.peak_ix_to_pos = file["1/peak_ix_to_pos"][self.pos_mask]
        self.peak_ix_to_len = file["1/peak_ix_to_len"][self.pos_mask]
        self.peak_ix_to_chr = file["1/peak_ix_to_chr"][self.pos_mask]

        
        pos_subset_mask = np.isin(self.peak_ix_to_chr.astype(str), subset) #only take the right (training, val) positions
        self.peak_ix_to_pos = self.peak_ix_to_pos[pos_subset_mask]
        self.peak_ix_to_len = self.peak_ix_to_len[pos_subset_mask]
        self.peak_ix_to_chr = self.peak_ix_to_chr[pos_subset_mask]

        self.mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
        self.rev_mapping = {v : k for k, v in self.mapping.items()}

        self.genome = {k : file["unstructured"][k] for k in list(file["unstructured"]) if k.startswith("chr")}
        self.file=file


    def __len__(self):
        return len(self.peak_ix_to_pos)*2
    

    def __getitem__(self, index):

        sample = {}
        sample["0/prot_names"] = self.TF
        sample["1/DNA_regions"] = []
        sample["central"] = []
        
        if index < len(self.peak_ix_to_pos): # take a positive
            chr = self.peak_ix_to_chr[index].decode()
            pos = self.peak_ix_to_pos[index]
            sample["1/DNA_regions"].append(self.genome[chr][pos-50:pos+51])
            sample["central"].append(1)

        else: # take a negative
        # get negative samples
            converted_index = index-len(self.peak_ix_to_pos) #convert the index!
            chr = self.peak_ix_to_chr[converted_index]
            pos = self.peak_ix_to_pos[converted_index]
            offset = np.random.choice([-1, 1]) * np.random.randint(101, 200) #! Should I double check to make sure that this is not also a positive region?
            temp_DNA = self.genome[chr][pos + offset - 50:pos + offset + 51]
            sample["1/DNA_regions"].append(temp_DNA)
            sample["central"].append(0)
        
        return sample





# class DataModule(pl.LightningDataModule):
#     def __init__(
#             self,
#             h5torch_file,
#             batch_size,
#             neg_mode
#     ):
#         super().__init__()

#         self.batch_size = batch_size
#         f = h5torch.File(h5torch_file) # .to_dict()
#         neg_modes = {
#             "neighbors": neighbor_negs,
#             "shuffled": shuffled_negs,
#             "dinucl_shuffled": dinucl_shuffled_negs,
#             "dinucl_sampled": dinucl_sampled_negs
#         }

#         if neg_mode not in neg_modes:
#             raise ValueError(f"Invalid neg_mode: {neg_mode}")
        
#         train_subset = ["chr1", "chr4", "chr7", "chr10", "chr13", "chr16", "chr19"] #! also include the alts!!!
#         val_subset = ["chr2", "chr5", "chr8", "chr11", "chr14", "chr17", "chr20"]

#         neg_class = neg_modes[neg_mode]
#         self.train_data = neg_class(f, subset=train_subset, num_samples=1000)
#         self.val_data = neg_class(f, subset=val_subset, num_samples=1000)


#     def setup(self, stage=None):
#         pass

#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(self.train_data, shuffle=True, batch_size = self.batch_size, num_workers = 10)
    
#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(self.val_data, shuffle=True, batch_size = self.batch_size, num_workers = 10)
