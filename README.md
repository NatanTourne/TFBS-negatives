### plmbind

## TODO
- [ ] Make TF splits, make sure they are used in plmbindDataModule
- [ ] "Hi-qual" test set -- write separate validation datasets and validation_step hooks
- [ ] Add loss weighing for dinucleotide negatives



### Generate dataset

```bash
python plmbind/utils/Generate_Dataset/Generate_Dataset.py --Generate_h5t --original_bed_file remap2022_crm_macs2_hg38_v1_0.bed --genome_file hg38.2bit --h5t_file remap.h5t

python plmbind/utils/Generate_Dataset/Generate_Dataset.py --Generate_esm2_embeddings --h5t_file remap.h5t --embeddings_model facebook/esm2_t6_8M_UR50D --used_GPU 0

python plmbind/utils/Generate_Dataset/Generate_Dataset.py --Generate_dinucl_shuffled_neg --h5t_file remap.h5t

...
```


### Minimal working script example

```python
from plmbind.data import plmbindDataModule
from plmbind.models import Plmbind
import pytorch_lightning as pl


balanced_proteins = True

dm = plmbindDataModule(
    "remap.h5t",
    batch_size=8,
    balance_proteins=balanced_proteins,
    protein_key = "esm2_t6_8M_UR50D",
    return_protein_embeddings = True,
    return_protein_seqs=False,
    negative_samples_mode="shuffle",
    n_negatives_per_positive=1,
)

model = Plmbind()

trainer = pl.Trainer(
    max_steps = 5_000,
    val_check_interval = 100,
    limit_val_batches=(None if balanced_proteins else 100), 
    accelerator = "gpu", 
    devices = [0],
)

trainer.fit(model, dm)
```