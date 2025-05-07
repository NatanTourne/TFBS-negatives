import h5torch
import numpy as np
import torch
import pytorch_lightning as pl
import warnings
from pytorch_lightning.callbacks import Callback
from itertools import permutations


class dinucl_shuffled_negs(h5torch.Dataset):
    def __init__(
        self,
        file,
        TF,
        subset=None
    ):
        super().__init__(file, in_memory=True)
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
        
        if index < len(self.peak_ix_to_pos): # take a positive
            chr = self.peak_ix_to_chr[index].decode()
            pos = self.peak_ix_to_pos[index]
            sample["1/DNA_regions"] = self.genome[chr][pos-50:pos+51]
            sample["central"] = 1

        else: # take a negative
        # get negative samples
            neg_seq = self.neg_seqs[index-len(self.peak_ix_to_pos)] #convert the index!
            sample["1/DNA_regions"] = neg_seq
            sample["central"] = 0
        
        return sample
    

class dinucl_sampled_negs(h5torch.Dataset):
    def __init__(
        self,
        file,
        TF,
        subset=None
    ):
        #! still have to test this!
        super().__init__(file, in_memory=True)
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
        
        if index < len(self.peak_ix_to_pos): # take a positive
            chr = self.peak_ix_to_chr[index].decode()
            pos = self.peak_ix_to_pos[index]
            sample["1/DNA_regions"] = self.genome[chr][pos-50:pos+51]
            sample["central"] = 1

        else: # take a negative
        # get negative samples
            converted_index = index-len(self.peak_ix_to_pos) #convert the index!
            chr = self.neg_chrs[converted_index].decode()
            pos = self.neg_pos[converted_index]
            sample["1/DNA_regions"] = self.genome[chr][pos-50:pos+51]
            sample["central"] = 0
        
        return sample

class shuffled_negs(h5torch.Dataset):
    def __init__(
        self,
        file,
        TF,
        subset=None
    ):
        super().__init__(file, in_memory=True)
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

        
        if index < len(self.peak_ix_to_pos): # take a positive
            chr = self.peak_ix_to_chr[index].decode()
            pos = self.peak_ix_to_pos[index]
            sample["1/DNA_regions"] = self.genome[chr][pos-50:pos+51]
            sample["central"] = 1

        else: # take a negative
        # get negative samples
            converted_index = index-len(self.peak_ix_to_pos) #convert the index!
            chr = self.peak_ix_to_chr[converted_index].decode()
            pos = self.peak_ix_to_pos[converted_index]
            DNA = self.genome[chr][pos-50:pos+51]
            shuffled_DNA =  np.random.permutation(DNA)
            sample["1/DNA_regions"] = shuffled_DNA
            sample["central"] = 0
        
        return sample
  
class neighbor_negs(h5torch.Dataset):
    def __init__(
        self,
        file,
        TF,
        subset=None
    ):
        #! still have to test this!
        super().__init__(file, in_memory=True)
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

        
        if index < len(self.peak_ix_to_pos): # take a positive
            chr = self.peak_ix_to_chr[index].decode()
            pos = self.peak_ix_to_pos[index]
            sample["1/DNA_regions"] = self.genome[chr][pos-50:pos+51]
            sample["central"] = 1

        else: # take a negative
        # get negative samples
            converted_index = index-len(self.peak_ix_to_pos) #convert the index!
            chr = self.peak_ix_to_chr[converted_index].decode()
            pos = self.peak_ix_to_pos[converted_index]
            offset = np.random.choice([-1, 1]) * np.random.randint(101, 200) #! Should I double check to make sure that this is not also a positive region?
            temp_DNA = self.genome[chr][pos + offset - 50:pos + offset + 51]
            sample["1/DNA_regions"] = temp_DNA
            sample["central"] = 0 
        
        return sample
    

class HQ_dataset(h5torch.Dataset):
    """
    This dataset returns the "High Quality" dataset. This means that for a given dataset (celltype) and TF,
    it returns the positive samples for that TF as positives and non overlapping ATAC peaks as negatives (also positives of other TFs?).
    Special care has to be taken to exclude positive samples from other TFs that partially overlap with the positive samples of the TF of interest.
    """
    def __init__(
        self,
        file,
        TF,
        subset=None
    ):
        super().__init__(file, in_memory=True)
        self.subset = subset
        self.TF = TF

        self.mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
        self.rev_mapping = {v : k for k, v in self.mapping.items()}

        self.genome = {k : file["unstructured"][k] for k in list(file["unstructured"]) if k.startswith("chr")}

        prot_names = file["0/prot_names"][:]
        prot_mask_ATAC = prot_names== b"ATAC_peak"
        prot_mask_TF = file["0/prot_names"][:] == TF.encode()
        central = file["central"][:]
        ATAC_c = central[prot_mask_ATAC]
        TF_c = central[prot_mask_TF]

        # put everything as 2 (not used)
        new_vector = np.full(ATAC_c.shape, 2)

        # set the positives from the TF as positives
        new_vector[(TF_c == 1)] = 1  # Set to 1 where TF_c is 1

        # if a position is open according to ATAC and it does not overlap with a positive of the TF take it as a negative
        new_vector[(ATAC_c == 1) & (TF_c != 2)] = 0 # Set to 0 where ATAC_c is 1 and TF_c is not 2

        # this one should not change anything!
        new_vector[(ATAC_c == 1) & (TF_c == 2)] = 2  # Ensure 2 where ATAC_c is 1 and TF_c is 2

        # now make a new central vector without the 2's and make the matching peak_ix_to_pos etc
        mask_atac = new_vector != 2
        mask_atac = mask_atac.squeeze()
        mask_subset = np.isin(file["1/peak_ix_to_chr"][:].astype(str), subset) #only take the right (training, val) positions
        mask = mask_atac & mask_subset

        self.central = new_vector[0].squeeze()[mask]
        self.peak_ix_to_pos = file["1/peak_ix_to_pos"][mask]
        self.peak_ix_to_len = file["1/peak_ix_to_len"][mask]
        self.peak_ix_to_chr = file["1/peak_ix_to_chr"][mask]

    def __len__(self):
        return len(self.peak_ix_to_pos)
    
    def __getitem__(self, index):
        sample = {}
        sample["0/prot_names"] = self.TF

        
        chr = self.peak_ix_to_chr[index].decode()
        pos = self.peak_ix_to_pos[index]
        #! BUT NOW WE HAVE TO DEAL WITH THE FACT THAT SOME POSITIONS ARE NOT 101BP
        len = self.peak_ix_to_len[index]
        if len == 101:
            sample["1/DNA_regions"] = self.genome[chr][pos-50:pos+51]
        elif len < 101:
            #! PLACEHOLDER JUST TAKING 101 bp FOR NOW
            sample["1/DNA_regions"] = self.genome[chr][pos-50:pos+51]
            warnings.warn("The handling of sequences with length < 101 is currently a placeholder and must be updated.")
        elif len > 101:
            #! WHAT IS THE BEST STRATEGY HERE? Different samples can be taken here
            sample["1/DNA_regions"] = self.genome[chr][pos-50:pos+51]
            warnings.warn("The handling of sequences with length > 101 is currently a placeholder and must be updated.")

        sample["central"] = self.central[index]

        return sample





class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            h5torch_file,
            TF,
            batch_size,
            neg_mode,
            cross_val_set = 0
    ):
        super().__init__()
        self.TF = TF
        self.batch_size = batch_size
        f = h5torch.File(h5torch_file) # .to_dict()
        neg_modes = {
            "neighbors": neighbor_negs,
            "shuffled": shuffled_negs,
            "dinucl_shuffled": dinucl_shuffled_negs,
            "dinucl_sampled": dinucl_sampled_negs
        }

        if neg_mode not in neg_modes:
            raise ValueError(f"Invalid neg_mode: {neg_mode}")
        
        #! todo, automatically get all the alts etc from the genome file and only give the main chrs here
        part1 =  ['chr13', 'chr13_KI270838v1_alt', 'chr13_KI270839v1_alt', 'chr13_KI270840v1_alt', 'chr13_KI270841v1_alt', 'chr13_KI270842v1_alt', 'chr13_KI270843v1_alt', 'chr18', 'chr18_GL383567v1_alt', 'chr18_GL383568v1_alt', 'chr18_GL383569v1_alt', 'chr18_GL383570v1_alt', 'chr18_GL383571v1_alt', 'chr18_GL383572v1_alt', 'chr18_KI270863v1_alt', 'chr18_KI270864v1_alt', 'chr18_KI270911v1_alt', 'chr18_KI270912v1_alt', 'chr19', 'chr19_GL000209v2_alt', 'chr19_GL383573v1_alt', 'chr19_GL383574v1_alt', 'chr19_GL383575v2_alt', 'chr19_GL383576v1_alt', 'chr19_GL949746v1_alt', 'chr19_GL949747v2_alt', 'chr19_GL949748v2_alt', 'chr19_GL949749v2_alt', 'chr19_GL949750v2_alt', 'chr19_GL949751v2_alt', 'chr19_GL949752v1_alt', 'chr19_GL949753v2_alt', 'chr19_KI270865v1_alt', 'chr19_KI270866v1_alt', 'chr19_KI270867v1_alt', 'chr19_KI270868v1_alt', 'chr19_KI270882v1_alt', 'chr19_KI270883v1_alt', 'chr19_KI270884v1_alt', 'chr19_KI270885v1_alt', 'chr19_KI270886v1_alt', 'chr19_KI270887v1_alt', 'chr19_KI270888v1_alt', 'chr19_KI270889v1_alt', 'chr19_KI270890v1_alt', 'chr19_KI270891v1_alt', 'chr19_KI270914v1_alt', 'chr19_KI270915v1_alt', 'chr19_KI270916v1_alt', 'chr19_KI270917v1_alt', 'chr19_KI270918v1_alt', 'chr19_KI270919v1_alt', 'chr19_KI270920v1_alt', 'chr19_KI270921v1_alt', 'chr19_KI270922v1_alt', 'chr19_KI270923v1_alt', 'chr19_KI270929v1_alt', 'chr19_KI270930v1_alt', 'chr19_KI270931v1_alt', 'chr19_KI270932v1_alt', 'chr19_KI270933v1_alt', 'chr19_KI270938v1_alt', 'chr20', 'chr20_GL383577v2_alt', 'chr20_KI270869v1_alt', 'chr20_KI270870v1_alt', 'chr20_KI270871v1_alt', 'chr3', 'chr3_GL000221v1_random', 'chr3_GL383526v1_alt', 'chr3_JH636055v2_alt', 'chr3_KI270777v1_alt', 'chr3_KI270778v1_alt', 'chr3_KI270779v1_alt', 'chr3_KI270780v1_alt', 'chr3_KI270781v1_alt', 'chr3_KI270782v1_alt', 'chr3_KI270783v1_alt', 'chr3_KI270784v1_alt', 'chr3_KI270895v1_alt', 'chr3_KI270924v1_alt', 'chr3_KI270934v1_alt', 'chr3_KI270935v1_alt', 'chr3_KI270936v1_alt', 'chr3_KI270937v1_alt', 'chr4', 'chr4_GL000008v2_random', 'chr4_GL000257v2_alt', 'chr4_GL383527v1_alt', 'chr4_GL383528v1_alt', 'chr4_KI270785v1_alt', 'chr4_KI270786v1_alt', 'chr4_KI270787v1_alt', 'chr4_KI270788v1_alt', 'chr4_KI270789v1_alt', 'chr4_KI270790v1_alt', 'chr4_KI270896v1_alt', 'chr4_KI270925v1_alt', 'chr7', 'chr7_GL383534v2_alt', 'chr7_KI270803v1_alt', 'chr7_KI270804v1_alt', 'chr7_KI270805v1_alt', 'chr7_KI270806v1_alt', 'chr7_KI270807v1_alt', 'chr7_KI270808v1_alt', 'chr7_KI270809v1_alt', 'chr7_KI270899v1_alt', 'chrX', 'chrX_KI270880v1_alt', 'chrX_KI270881v1_alt', 'chrX_KI270913v1_alt']
        part2 = ['chr1', 'chr10', 'chr10_GL383545v1_alt', 'chr10_GL383546v1_alt', 'chr10_KI270824v1_alt', 'chr10_KI270825v1_alt', 'chr11', 'chr11_GL383547v1_alt', 'chr11_JH159136v1_alt', 'chr11_JH159137v1_alt', 'chr11_KI270721v1_random', 'chr11_KI270826v1_alt', 'chr11_KI270827v1_alt', 'chr11_KI270829v1_alt', 'chr11_KI270830v1_alt', 'chr11_KI270831v1_alt', 'chr11_KI270832v1_alt', 'chr11_KI270902v1_alt', 'chr11_KI270903v1_alt', 'chr11_KI270927v1_alt', 'chr15', 'chr15_GL383554v1_alt', 'chr15_GL383555v2_alt', 'chr15_KI270727v1_random', 'chr15_KI270848v1_alt', 'chr15_KI270849v1_alt', 'chr15_KI270850v1_alt', 'chr15_KI270851v1_alt', 'chr15_KI270852v1_alt', 'chr15_KI270905v1_alt', 'chr15_KI270906v1_alt', 'chr1_GL383518v1_alt', 'chr1_GL383519v1_alt', 'chr1_GL383520v2_alt', 'chr1_KI270706v1_random', 'chr1_KI270707v1_random', 'chr1_KI270708v1_random', 'chr1_KI270709v1_random', 'chr1_KI270710v1_random', 'chr1_KI270711v1_random', 'chr1_KI270712v1_random', 'chr1_KI270713v1_random', 'chr1_KI270714v1_random', 'chr1_KI270759v1_alt', 'chr1_KI270760v1_alt', 'chr1_KI270761v1_alt', 'chr1_KI270762v1_alt', 'chr1_KI270763v1_alt', 'chr1_KI270764v1_alt', 'chr1_KI270765v1_alt', 'chr1_KI270766v1_alt', 'chr1_KI270892v1_alt', 'chr21', 'chr21_GL383578v2_alt', 'chr21_GL383579v2_alt', 'chr21_GL383580v2_alt', 'chr21_GL383581v2_alt', 'chr21_KI270872v1_alt', 'chr21_KI270873v1_alt', 'chr21_KI270874v1_alt', 'chr22', 'chr22_GL383582v2_alt', 'chr22_GL383583v2_alt', 'chr22_KB663609v1_alt', 'chr22_KI270731v1_random', 'chr22_KI270732v1_random', 'chr22_KI270733v1_random', 'chr22_KI270734v1_random', 'chr22_KI270735v1_random', 'chr22_KI270736v1_random', 'chr22_KI270737v1_random', 'chr22_KI270738v1_random', 'chr22_KI270739v1_random', 'chr22_KI270875v1_alt', 'chr22_KI270876v1_alt', 'chr22_KI270877v1_alt', 'chr22_KI270878v1_alt', 'chr22_KI270879v1_alt', 'chr22_KI270928v1_alt', 'chr9', 'chr9_GL383539v1_alt', 'chr9_GL383540v1_alt', 'chr9_GL383541v1_alt', 'chr9_GL383542v1_alt', 'chr9_KI270717v1_random', 'chr9_KI270718v1_random', 'chr9_KI270719v1_random', 'chr9_KI270720v1_random', 'chr9_KI270823v1_alt', 'chrY', 'chrY_KI270740v1_random']
        part3 = ['chr12', 'chr12_GL383549v1_alt', 'chr12_GL383550v2_alt', 'chr12_GL383551v1_alt', 'chr12_GL383552v1_alt', 'chr12_GL383553v2_alt', 'chr12_GL877875v1_alt', 'chr12_GL877876v1_alt', 'chr12_KI270833v1_alt', 'chr12_KI270834v1_alt', 'chr12_KI270835v1_alt', 'chr12_KI270836v1_alt', 'chr12_KI270837v1_alt', 'chr12_KI270904v1_alt', 'chr14', 'chr14_GL000009v2_random', 'chr14_GL000194v1_random', 'chr14_GL000225v1_random', 'chr14_KI270722v1_random', 'chr14_KI270723v1_random', 'chr14_KI270724v1_random', 'chr14_KI270725v1_random', 'chr14_KI270726v1_random', 'chr14_KI270844v1_alt', 'chr14_KI270845v1_alt', 'chr14_KI270846v1_alt', 'chr14_KI270847v1_alt', 'chr16', 'chr16_GL383556v1_alt', 'chr16_GL383557v1_alt', 'chr16_KI270728v1_random', 'chr16_KI270853v1_alt', 'chr16_KI270854v1_alt', 'chr16_KI270855v1_alt', 'chr16_KI270856v1_alt', 'chr17', 'chr17_GL000205v2_random', 'chr17_GL000258v2_alt', 'chr17_GL383563v3_alt', 'chr17_GL383564v2_alt', 'chr17_GL383565v1_alt', 'chr17_GL383566v1_alt', 'chr17_JH159146v1_alt', 'chr17_JH159147v1_alt', 'chr17_JH159148v1_alt', 'chr17_KI270729v1_random', 'chr17_KI270730v1_random', 'chr17_KI270857v1_alt', 'chr17_KI270858v1_alt', 'chr17_KI270859v1_alt', 'chr17_KI270860v1_alt', 'chr17_KI270861v1_alt', 'chr17_KI270862v1_alt', 'chr17_KI270907v1_alt', 'chr17_KI270908v1_alt', 'chr17_KI270909v1_alt', 'chr17_KI270910v1_alt', 'chr2', 'chr2_GL383521v1_alt', 'chr2_GL383522v1_alt', 'chr2_GL582966v2_alt', 'chr2_KI270715v1_random', 'chr2_KI270716v1_random', 'chr2_KI270767v1_alt', 'chr2_KI270768v1_alt', 'chr2_KI270769v1_alt', 'chr2_KI270770v1_alt', 'chr2_KI270771v1_alt', 'chr2_KI270772v1_alt', 'chr2_KI270773v1_alt', 'chr2_KI270774v1_alt', 'chr2_KI270775v1_alt', 'chr2_KI270776v1_alt', 'chr2_KI270893v1_alt', 'chr2_KI270894v1_alt', 'chr5', 'chr5_GL000208v1_random', 'chr5_GL339449v2_alt', 'chr5_GL383530v1_alt', 'chr5_GL383531v1_alt', 'chr5_GL383532v1_alt', 'chr5_GL949742v1_alt', 'chr5_KI270791v1_alt', 'chr5_KI270792v1_alt', 'chr5_KI270793v1_alt', 'chr5_KI270794v1_alt', 'chr5_KI270795v1_alt', 'chr5_KI270796v1_alt', 'chr5_KI270897v1_alt', 'chr5_KI270898v1_alt', 'chr6', 'chr6_GL000250v2_alt', 'chr6_GL000251v2_alt', 'chr6_GL000252v2_alt', 'chr6_GL000253v2_alt', 'chr6_GL000254v2_alt', 'chr6_GL000255v2_alt', 'chr6_GL000256v2_alt', 'chr6_GL383533v1_alt', 'chr6_KB021644v2_alt', 'chr6_KI270758v1_alt', 'chr6_KI270797v1_alt', 'chr6_KI270798v1_alt', 'chr6_KI270799v1_alt', 'chr6_KI270800v1_alt', 'chr6_KI270801v1_alt', 'chr6_KI270802v1_alt', 'chr8', 'chr8_KI270810v1_alt', 'chr8_KI270811v1_alt', 'chr8_KI270812v1_alt', 'chr8_KI270813v1_alt', 'chr8_KI270814v1_alt', 'chr8_KI270815v1_alt', 'chr8_KI270816v1_alt', 'chr8_KI270817v1_alt', 'chr8_KI270818v1_alt', 'chr8_KI270819v1_alt', 'chr8_KI270820v1_alt', 'chr8_KI270821v1_alt', 'chr8_KI270822v1_alt', 'chr8_KI270900v1_alt', 'chr8_KI270901v1_alt', 'chr8_KI270926v1_alt']
        parts = [part1, part2, part3]
        all_orderings = list(permutations(parts))
        all_orderings = [list(ordering) for ordering in all_orderings]

        neg_class = neg_modes[neg_mode]
        # depending on the cross_val_set, take different subsets of the data for train, val, test
        self.train_data = neg_class(f, TF=TF, subset=all_orderings[cross_val_set][0])
        self.val_data = neg_class(f, TF=TF, subset=all_orderings[cross_val_set][1])
        self.HQ_val_data = HQ_dataset(f, TF=TF, subset=all_orderings[cross_val_set][1])


    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, shuffle=True, batch_size = self.batch_size, num_workers = 2)
    
    def val_dataloader(self):
        return [torch.utils.data.DataLoader(self.val_data, shuffle=True, batch_size=self.batch_size, num_workers=2), torch.utils.data.DataLoader(self.HQ_val_data, shuffle=False, batch_size=self.batch_size, num_workers=2)]


class DataModule_sanity_check(DataModule):
    def val_dataloader(self):
        warnings.warn("Sanity check mode: Validation is being calculated on training data.")
        return torch.utils.data.DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size, num_workers=2)