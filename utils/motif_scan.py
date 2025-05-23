import os
import h5torch
import numpy as np
from pyjaspar import jaspardb
import pandas as pd
from Bio.Seq import Seq
from TFBS_negatives.data import HQ_dataset
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd

out_folder = "/data/home/natant/Negatives/Runs/full_run_1/motifs"
data_folder = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr/"
h5t_files = [f for f in os.listdir(data_folder) if f.endswith('.h5t')]
prot_names = []
for h5t_file in h5t_files:
    file_path = os.path.join(data_folder, h5t_file)
    file = h5torch.File(file_path, 'r')
    
    prot_names.extend(file["0/prot_names"][:].astype(str).tolist())

unique_tfs = np.unique(prot_names)
unique_tfs = unique_tfs[unique_tfs != "ATAC_peak"]

jaspar_connection = """ARID3A_(NB100-279):
ATF1_(06-325):
ATF2_(SC-81188): MA1632.1 MA1632.2
ATF3: MA0605.2 MA0605.3
BHLHE40: MA0464.2 MA0464.3
Bach1_(sc-14700): MA1633.1 MA1633.2
CEBPB_(SC-150): MA0466.1 MA0466.2 MA0466.3 MA0466.4
CEBPD_(SC-636): MA0836.1 MA0836.2 MA0836.3
CREB1_(SC-240): MA0018.1 MA0018.2 MA0018.3 MA0018.4 MA0018.5
CTCF: MA0139.1 MA0139.2 MA1929.1 MA1929.2 MA1930.1 MA1930.2
ELF1_(SC-631): MA0473.1 MA0473.2 MA0473.3 MA0473.4
ELK1_(1277-1): MA0028.1 MA0028.2 MA0028.3
ETS1: MA0098.1 MA0098.3 MA0098.4
Egr-1: MA0162.2 MA0162.3 MA0162.4 MA0162.5
FOSL1_(SC-183): MA0477.1 MA0477.2 MA0477.3
FOSL2: MA0478.1 MA0478.2
FOXA1_(SC-101058): MA0148.1 MA0148.2 MA0148.3 MA0148.4 MA0148.5
FOXM1_(SC-502): UN0802.1
GATA3_(SC-268): MA0037.1 MA0037.2 MA0037.3
HSF1: MA0486.1 MA0486.2
IKZF1_(IkN)_(UCLA): MA1508.1 MA1508.2
IRF3: MA1418.1 MA1418.2
JunD: MA0491.1 MA0491.2 MA0491.3 MA0492.1 MA0492.2
MAZ_(ab85725): MA1522.1 MA1522.2
MEF2A: MA0052.1 MA0052.2 MA0052.3 MA0052.4 MA0052.5
MYBL2_(SC-81192): MA0777.1
MafF_(M8194): MA0495.1 MA0495.2 MA0495.3 MA0495.4
MafK_(ab50322): MA0496.1 MA0496.2 MA0496.3 MA0496.4
Max: MA0058.1 MA0058.2 MA0058.3 MA0058.4
Mxi1_(AF4185): MA1108.1 MA1108.2 MA1108.3
NF-YA: MA0060.1 MA0060.2 MA0060.3 MA0060.4
NF-YB: MA0502.1 MA0502.2 MA0502.3
NFIC_(SC-81335): MA0161.1 MA0161.2 MA0161.3 MA1527.1 MA1527.2
NR2F2_(SC-271940): MA1111.1 MA1111.2
Nrf1: MA0506.1
Pbx3: MA1114.1 MA1114.2
RFX5_(200-401-194): MA0510.1 MA0510.2 MA0510.3
RXRA:
SETDB1:
SIX5:
SP1: MA0079.1 MA0079.2 MA0079.3 MA0079.4 MA0079.5
SRF: MA0083.1 MA0083.2 MA0083.3
STAT5A_(SC-74442):
TBP:
TCF12: MA1648.1 MA1648.2
TCF7L2: MA0523.1 MA0523.2
TEAD4_(SC-101184): MA0809.1 MA0809.2 MA0809.3
USF-1:
USF2: MA0526.1 MA0526.2 MA0526.3 MA0526.4 MA0526.5
YY1_(SC-281): MA0095.1 MA0095.2
ZBTB33: MA0527.1 MA0527.2
ZBTB7A_(SC-34508): MA0750.1 MA0750.2 MA0750.3
ZEB1_(SC-25388): MA0103.2 MA0103.3 MA0103.4
ZNF217:
ZNF274: MA1592.1 MA1592.2
ZZZ3:
Znf143_(16618-1-AP): MA0088.2
"""
jaspar_dict = {}
for line in jaspar_connection.strip().split("\n"):
    if ":" in line:
        tf, matrices = line.split(":", 1)
        jaspar_dict[tf.strip()] = matrices.strip().split() if matrices.strip() else []

matrix_ids = [value for values in jaspar_dict.values() for value in values]

jdb_obj = jaspardb(release='JASPAR2024')
motif_objects = {}
for mid in matrix_ids:
    motif = jdb_obj.fetch_motif_by_id(mid)
    motif.pseudocounts = 0.8
    pssm = motif.pssm  # compute the position-specific scoring matrix (PSSM)
    motif_objects[mid] = pssm

part1 =  ['chr13', 'chr13_KI270838v1_alt', 'chr13_KI270839v1_alt', 'chr13_KI270840v1_alt', 'chr13_KI270841v1_alt', 'chr13_KI270842v1_alt', 'chr13_KI270843v1_alt', 'chr18', 'chr18_GL383567v1_alt', 'chr18_GL383568v1_alt', 'chr18_GL383569v1_alt', 'chr18_GL383570v1_alt', 'chr18_GL383571v1_alt', 'chr18_GL383572v1_alt', 'chr18_KI270863v1_alt', 'chr18_KI270864v1_alt', 'chr18_KI270911v1_alt', 'chr18_KI270912v1_alt', 'chr19', 'chr19_GL000209v2_alt', 'chr19_GL383573v1_alt', 'chr19_GL383574v1_alt', 'chr19_GL383575v2_alt', 'chr19_GL383576v1_alt', 'chr19_GL949746v1_alt', 'chr19_GL949747v2_alt', 'chr19_GL949748v2_alt', 'chr19_GL949749v2_alt', 'chr19_GL949750v2_alt', 'chr19_GL949751v2_alt', 'chr19_GL949752v1_alt', 'chr19_GL949753v2_alt', 'chr19_KI270865v1_alt', 'chr19_KI270866v1_alt', 'chr19_KI270867v1_alt', 'chr19_KI270868v1_alt', 'chr19_KI270882v1_alt', 'chr19_KI270883v1_alt', 'chr19_KI270884v1_alt', 'chr19_KI270885v1_alt', 'chr19_KI270886v1_alt', 'chr19_KI270887v1_alt', 'chr19_KI270888v1_alt', 'chr19_KI270889v1_alt', 'chr19_KI270890v1_alt', 'chr19_KI270891v1_alt', 'chr19_KI270914v1_alt', 'chr19_KI270915v1_alt', 'chr19_KI270916v1_alt', 'chr19_KI270917v1_alt', 'chr19_KI270918v1_alt', 'chr19_KI270919v1_alt', 'chr19_KI270920v1_alt', 'chr19_KI270921v1_alt', 'chr19_KI270922v1_alt', 'chr19_KI270923v1_alt', 'chr19_KI270929v1_alt', 'chr19_KI270930v1_alt', 'chr19_KI270931v1_alt', 'chr19_KI270932v1_alt', 'chr19_KI270933v1_alt', 'chr19_KI270938v1_alt', 'chr20', 'chr20_GL383577v2_alt', 'chr20_KI270869v1_alt', 'chr20_KI270870v1_alt', 'chr20_KI270871v1_alt', 'chr3', 'chr3_GL000221v1_random', 'chr3_GL383526v1_alt', 'chr3_JH636055v2_alt', 'chr3_KI270777v1_alt', 'chr3_KI270778v1_alt', 'chr3_KI270779v1_alt', 'chr3_KI270780v1_alt', 'chr3_KI270781v1_alt', 'chr3_KI270782v1_alt', 'chr3_KI270783v1_alt', 'chr3_KI270784v1_alt', 'chr3_KI270895v1_alt', 'chr3_KI270924v1_alt', 'chr3_KI270934v1_alt', 'chr3_KI270935v1_alt', 'chr3_KI270936v1_alt', 'chr3_KI270937v1_alt', 'chr4', 'chr4_GL000008v2_random', 'chr4_GL000257v2_alt', 'chr4_GL383527v1_alt', 'chr4_GL383528v1_alt', 'chr4_KI270785v1_alt', 'chr4_KI270786v1_alt', 'chr4_KI270787v1_alt', 'chr4_KI270788v1_alt', 'chr4_KI270789v1_alt', 'chr4_KI270790v1_alt', 'chr4_KI270896v1_alt', 'chr4_KI270925v1_alt', 'chr7', 'chr7_GL383534v2_alt', 'chr7_KI270803v1_alt', 'chr7_KI270804v1_alt', 'chr7_KI270805v1_alt', 'chr7_KI270806v1_alt', 'chr7_KI270807v1_alt', 'chr7_KI270808v1_alt', 'chr7_KI270809v1_alt', 'chr7_KI270899v1_alt', 'chrX', 'chrX_KI270880v1_alt', 'chrX_KI270881v1_alt', 'chrX_KI270913v1_alt']
part2 = ['chr1', 'chr10', 'chr10_GL383545v1_alt', 'chr10_GL383546v1_alt', 'chr10_KI270824v1_alt', 'chr10_KI270825v1_alt', 'chr11', 'chr11_GL383547v1_alt', 'chr11_JH159136v1_alt', 'chr11_JH159137v1_alt', 'chr11_KI270721v1_random', 'chr11_KI270826v1_alt', 'chr11_KI270827v1_alt', 'chr11_KI270829v1_alt', 'chr11_KI270830v1_alt', 'chr11_KI270831v1_alt', 'chr11_KI270832v1_alt', 'chr11_KI270902v1_alt', 'chr11_KI270903v1_alt', 'chr11_KI270927v1_alt', 'chr15', 'chr15_GL383554v1_alt', 'chr15_GL383555v2_alt', 'chr15_KI270727v1_random', 'chr15_KI270848v1_alt', 'chr15_KI270849v1_alt', 'chr15_KI270850v1_alt', 'chr15_KI270851v1_alt', 'chr15_KI270852v1_alt', 'chr15_KI270905v1_alt', 'chr15_KI270906v1_alt', 'chr1_GL383518v1_alt', 'chr1_GL383519v1_alt', 'chr1_GL383520v2_alt', 'chr1_KI270706v1_random', 'chr1_KI270707v1_random', 'chr1_KI270708v1_random', 'chr1_KI270709v1_random', 'chr1_KI270710v1_random', 'chr1_KI270711v1_random', 'chr1_KI270712v1_random', 'chr1_KI270713v1_random', 'chr1_KI270714v1_random', 'chr1_KI270759v1_alt', 'chr1_KI270760v1_alt', 'chr1_KI270761v1_alt', 'chr1_KI270762v1_alt', 'chr1_KI270763v1_alt', 'chr1_KI270764v1_alt', 'chr1_KI270765v1_alt', 'chr1_KI270766v1_alt', 'chr1_KI270892v1_alt', 'chr21', 'chr21_GL383578v2_alt', 'chr21_GL383579v2_alt', 'chr21_GL383580v2_alt', 'chr21_GL383581v2_alt', 'chr21_KI270872v1_alt', 'chr21_KI270873v1_alt', 'chr21_KI270874v1_alt', 'chr22', 'chr22_GL383582v2_alt', 'chr22_GL383583v2_alt', 'chr22_KB663609v1_alt', 'chr22_KI270731v1_random', 'chr22_KI270732v1_random', 'chr22_KI270733v1_random', 'chr22_KI270734v1_random', 'chr22_KI270735v1_random', 'chr22_KI270736v1_random', 'chr22_KI270737v1_random', 'chr22_KI270738v1_random', 'chr22_KI270739v1_random', 'chr22_KI270875v1_alt', 'chr22_KI270876v1_alt', 'chr22_KI270877v1_alt', 'chr22_KI270878v1_alt', 'chr22_KI270879v1_alt', 'chr22_KI270928v1_alt', 'chr9', 'chr9_GL383539v1_alt', 'chr9_GL383540v1_alt', 'chr9_GL383541v1_alt', 'chr9_GL383542v1_alt', 'chr9_KI270717v1_random', 'chr9_KI270718v1_random', 'chr9_KI270719v1_random', 'chr9_KI270720v1_random', 'chr9_KI270823v1_alt', 'chrY', 'chrY_KI270740v1_random']
part3 = ['chr12', 'chr12_GL383549v1_alt', 'chr12_GL383550v2_alt', 'chr12_GL383551v1_alt', 'chr12_GL383552v1_alt', 'chr12_GL383553v2_alt', 'chr12_GL877875v1_alt', 'chr12_GL877876v1_alt', 'chr12_KI270833v1_alt', 'chr12_KI270834v1_alt', 'chr12_KI270835v1_alt', 'chr12_KI270836v1_alt', 'chr12_KI270837v1_alt', 'chr12_KI270904v1_alt', 'chr14', 'chr14_GL000009v2_random', 'chr14_GL000194v1_random', 'chr14_GL000225v1_random', 'chr14_KI270722v1_random', 'chr14_KI270723v1_random', 'chr14_KI270724v1_random', 'chr14_KI270725v1_random', 'chr14_KI270726v1_random', 'chr14_KI270844v1_alt', 'chr14_KI270845v1_alt', 'chr14_KI270846v1_alt', 'chr14_KI270847v1_alt', 'chr16', 'chr16_GL383556v1_alt', 'chr16_GL383557v1_alt', 'chr16_KI270728v1_random', 'chr16_KI270853v1_alt', 'chr16_KI270854v1_alt', 'chr16_KI270855v1_alt', 'chr16_KI270856v1_alt', 'chr17', 'chr17_GL000205v2_random', 'chr17_GL000258v2_alt', 'chr17_GL383563v3_alt', 'chr17_GL383564v2_alt', 'chr17_GL383565v1_alt', 'chr17_GL383566v1_alt', 'chr17_JH159146v1_alt', 'chr17_JH159147v1_alt', 'chr17_JH159148v1_alt', 'chr17_KI270729v1_random', 'chr17_KI270730v1_random', 'chr17_KI270857v1_alt', 'chr17_KI270858v1_alt', 'chr17_KI270859v1_alt', 'chr17_KI270860v1_alt', 'chr17_KI270861v1_alt', 'chr17_KI270862v1_alt', 'chr17_KI270907v1_alt', 'chr17_KI270908v1_alt', 'chr17_KI270909v1_alt', 'chr17_KI270910v1_alt', 'chr2', 'chr2_GL383521v1_alt', 'chr2_GL383522v1_alt', 'chr2_GL582966v2_alt', 'chr2_KI270715v1_random', 'chr2_KI270716v1_random', 'chr2_KI270767v1_alt', 'chr2_KI270768v1_alt', 'chr2_KI270769v1_alt', 'chr2_KI270770v1_alt', 'chr2_KI270771v1_alt', 'chr2_KI270772v1_alt', 'chr2_KI270773v1_alt', 'chr2_KI270774v1_alt', 'chr2_KI270775v1_alt', 'chr2_KI270776v1_alt', 'chr2_KI270893v1_alt', 'chr2_KI270894v1_alt', 'chr5', 'chr5_GL000208v1_random', 'chr5_GL339449v2_alt', 'chr5_GL383530v1_alt', 'chr5_GL383531v1_alt', 'chr5_GL383532v1_alt', 'chr5_GL949742v1_alt', 'chr5_KI270791v1_alt', 'chr5_KI270792v1_alt', 'chr5_KI270793v1_alt', 'chr5_KI270794v1_alt', 'chr5_KI270795v1_alt', 'chr5_KI270796v1_alt', 'chr5_KI270897v1_alt', 'chr5_KI270898v1_alt', 'chr6', 'chr6_GL000250v2_alt', 'chr6_GL000251v2_alt', 'chr6_GL000252v2_alt', 'chr6_GL000253v2_alt', 'chr6_GL000254v2_alt', 'chr6_GL000255v2_alt', 'chr6_GL000256v2_alt', 'chr6_GL383533v1_alt', 'chr6_KB021644v2_alt', 'chr6_KI270758v1_alt', 'chr6_KI270797v1_alt', 'chr6_KI270798v1_alt', 'chr6_KI270799v1_alt', 'chr6_KI270800v1_alt', 'chr6_KI270801v1_alt', 'chr6_KI270802v1_alt', 'chr8', 'chr8_KI270810v1_alt', 'chr8_KI270811v1_alt', 'chr8_KI270812v1_alt', 'chr8_KI270813v1_alt', 'chr8_KI270814v1_alt', 'chr8_KI270815v1_alt', 'chr8_KI270816v1_alt', 'chr8_KI270817v1_alt', 'chr8_KI270818v1_alt', 'chr8_KI270819v1_alt', 'chr8_KI270820v1_alt', 'chr8_KI270821v1_alt', 'chr8_KI270822v1_alt', 'chr8_KI270900v1_alt', 'chr8_KI270901v1_alt', 'chr8_KI270926v1_alt']
parts = [part1, part2, part3]

cell_types = ["MCF-7", "K562", "GM12878", "HepG2", "HEK293", "A549"]


for celltype in cell_types:
    AUROC_scores = {}
    accuracy_scores = {}
    print("celltype: " + celltype)
    for fold in range(3):
        print("fold: " + str(fold))
        file = h5torch.File(data_folder+celltype+".h5t", 'r')
        TF_list = [TF.decode() for TF in file["0/prot_names"][:]]
        TF_list.remove("ATAC_peak")

        results = {}
        true_vals = {}
        for TF in TF_list:
            dataset = HQ_dataset(file, TF, subset=parts[fold])
            print(f"TF: {TF}, length: {dataset.__len__()}")
            matrices = jaspar_dict[TF]
            true_vals[TF] = []
            if matrices == []:
                print(f"No matrices found for {TF}")
                continue
            else:
                print(f"Found {len(matrices)} matrices for {TF}")
                for i in range(dataset.__len__()): 
                    # Scan each PWM on both strands
                    seq = Seq("".join([dataset.rev_mapping[i] for i in dataset.__getitem__(i)["1/DNA_regions"]]))
                    true_vals[TF].append(dataset.__getitem__(i)["central"])
                    
                    best_score = []
                    for mid in matrices:
                        pssm = motif_objects[mid] 
                        # Score forward strand
                        scores_fwd = pssm.calculate(seq)
                        max_fwd = np.nanmax(scores_fwd) if len(scores_fwd)>0 else float('-inf')
                        # Score reverse complement
                        rc_seq = str(Seq(seq).reverse_complement())
                        scores_rev = pssm.calculate(rc_seq)
                        max_rev = np.nanmax(scores_rev) if len(scores_rev)>0 else float('-inf')
                        # Take the best (highest) score
                        best_score.append(np.nanmax([max_fwd, max_rev]))

                    results.setdefault(TF, []).append(np.nanmax(best_score))


        for TF, scores in results.items():
            labels = true_vals[TF]
            scores = np.array(scores)
            labels = np.array(labels)
            
            if len(scores) != len(labels):
                print(f"Length mismatch for {TF}: {len(scores)} vs {len(labels)}")
                accuracy_scores[TF] = None
                AUROC_scores[TF] = None
                continue
            if len(np.unique(labels)) < 2:
                print(f"Only one class present for {TF}")
                accuracy_scores[TF] = None
                AUROC_scores[TF] = None
                continue
            if len(scores) == 0:
                print(f"No scores for {TF}")
                accuracy_scores[TF] = None
                AUROC_scores[TF] = None
                continue
            if len(labels) == 0:
                print(f"No labels for {TF}")
                accuracy_scores[TF] = None
                AUROC_scores[TF] = None
                continue
            if np.isnan(scores).any():
                print(f"NaN values in scores for {TF}")
                accuracy_scores[TF] = None
                AUROC_scores[TF] = None
                continue
            if np.isnan(labels).any():
                print(f"NaN values in labels for {TF}")
                accuracy_scores[TF] = None
                AUROC_scores[TF] = None
                continue
            if len(np.unique(labels)) == 1:
                print(f"Only one class present in labels for {TF}")
                accuracy_scores[TF] = None
                AUROC_scores[TF] = None
                continue
            
            # Calculate AUROC score
            try:
                auc = roc_auc_score(labels, scores)
                AUROC_scores.setdefault(TF, []).append(auc)
            except ValueError as e:
                print(f"ValueError for {TF}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error for {TF}: {e}")
                continue
            
            # Calculate accuracy score with threshold at 0.8 * max_score
            try:
                max_score = np.max(scores)
                threshold = 0.8 * max_score #! important setting for accuracy
                predictions = (scores > threshold).astype(int)
                acc = accuracy_score(labels, predictions)
                accuracy_scores.setdefault(TF, []).append(acc)
            except Exception as e:
                print(f"Error calculating accuracy for {TF}: {e}")
                continue

        


    df = pd.DataFrame.from_dict(
        {tf: AUROC_scores[tf] + accuracy_scores[tf] for tf in AUROC_scores.keys()},
        orient='index',
        columns=["AUROC_1", "AUROC_2", "AUROC_3", "Accuracy_1", "Accuracy_2", "Accuracy_3"]
    )
    # Ensure the output folder exists
    os.makedirs(out_folder, exist_ok=True)

    # Write the dataframe to a CSV file in the specified output folder
    df.to_csv(os.path.join(out_folder, f"{celltype}.csv"), index=True)

