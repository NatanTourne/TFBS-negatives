from torch import nn, optim
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from transformers import EsmModel, EsmTokenizer
import warnings
from functools import partial
import numpy as np
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from torchmetrics.functional.classification import multilabel_average_precision, binary_auroc, binary_average_precision, binary_accuracy

#! OLD FUNCTIONS:
class MultiLabelLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y.float())
    
class MultiClassLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, y_hat, y):
        return F.cross_entropy(y_hat, torch.zeros(len(y_hat),dtype=torch.long))


class Permute(nn.Module): 
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x):
        return x.permute(*self.args)
    
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

def ConvTransposeBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.ConvTranspose1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )
    
class GlobalPool(nn.Module):
    def __init__(self, pooled_axis = 1, mode = "max"):
        super().__init__()
        assert mode in ["max", "mean"], "Only max and mean-pooling are implemented"
        if mode == "max":
            self.op = lambda x: torch.max(x, axis = pooled_axis).values
        elif mode == "mean":
            self.op = lambda x: torch.mean(x, axis = pooled_axis).values
    def forward(self, x):
        return self.op(x)

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)

class SmallMLP(nn.Module):
    def __init__(self, in_size, out_size, dropout = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, out_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_size * 4, out_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_size * 2, out_size),
        )

    def forward(self, x):
        return self.net(x)
    

class EnformerConvStack(nn.Module):
    def __init__(
        self,
        input_hsize = 4, # ..
        target_hsize = 512, # the last conv will have this many channels
        n_blocks = 6, # how many conv blocks there will be in the conv tower
        kernel_size = 9, # kernel size of the conv blocks, higher than in Enformer because we will not rely on a transformer
        latent_size = 64, # ..
        dropout=0.15, # ...
        progressive_channel_widening=True, # see docstring
        pooling_between_blocks=True # see docstring
        ):
        """
        This model structure follows the Enformer model stack, up until the transformer part (after that it's custom):
        https://www.nature.com/articles/s41592-021-01252-x/figures/5
        To get an output, Global max pooling followed with [Linear -> GeLU -> Dropout -> Linear] x 2

        Two optional modifications are possible:
        `progressive_channel_widening` =
            Enformer approach is to increase channel dims per block in conv tower. Can be put to false to just have a constant channel dim = target_hsize
        `pooling_between_blocks` =
            Enformer approach is to pool in each block in the conv tower. Can be put to False.
                
        Code inspired and taken from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py.
        """
        super().__init__()

        if progressive_channel_widening:
            factor = (target_hsize / (target_hsize // 2))**(1/n_blocks)
            hidden_sizes = [target_hsize // 2]
            for _ in range(n_blocks-1):
                hidden_sizes.append(int(hidden_sizes[-1] * factor))
            hidden_sizes.append(target_hsize)
        else:
            hidden_sizes = [target_hsize] * n_blocks

        layers = [
            nn.Conv1d(input_hsize, hidden_sizes[0], 15, padding = 15 // 2),
            Residual(ConvBlock(hidden_sizes[0], kernel_size = 1)),
        ]
        if pooling_between_blocks:
            layers.append(AttentionPool(hidden_sizes[0]))

            
        for i in range(1, len(hidden_sizes)):
            layers.append(ConvBlock(hidden_sizes[i-1], dim_out = hidden_sizes[i], kernel_size = kernel_size))
            layers.append(Residual(ConvBlock(hidden_sizes[i], kernel_size = 1)))
            if ((i+1) != len(hidden_sizes)) and pooling_between_blocks:
                layers.append(AttentionPool(hidden_sizes[i]))
        
        layers.append(GlobalPool(pooled_axis=2))
        layers.append(SmallMLP(target_hsize, latent_size, dropout = dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class PrintMod(nn.Module):
    def __init__(self):
        super(PrintMod, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

#! OLD model!!
class multilabel(pl.LightningModule):
    def __init__(
        self,

        # Model architecture
        target_hsize_DNA=64,
        n_blocks_DNA=2,
        DNA_kernel_size=9,
        progressive_channel_widening_DNA=True,
        pooling_between_blocks_DNA=True,
        DNA_dropout=0.25,
        latent_vector_size=64,

        # training
        learning_rate_DNA_branch=1e-5,
        loss_function = "MultiLabel",
        HQ_val_interval = 1,
    ):
        super(multilabel, self).__init__()

        self.learning_rate_DNA_branch = learning_rate_DNA_branch
        self.HQ_val_interval = HQ_val_interval
        self.val_step_output = []
        self.latent_vector_size = latent_vector_size

        # Setup Loss Functions (not all loss functions are possible for the validation loop)
        loss_function_options = ["MultiClass", "MultiLabel"]
        if loss_function not in loss_function_options:
            raise ValueError("Invalid loss function. Expected one of: %s" % loss_function_options)
        
        if loss_function == "MultiClass":
            self.loss_function = MultiClassLoss()
        elif loss_function == "MultiLabel":
            self.loss_function = MultiLabelLoss()

        # nucleotide encodings
        nucleotide_weights = torch.FloatTensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0]]
            )
        self.embedding = nn.Embedding.from_pretrained(nucleotide_weights)
        
        # DNA branch
    
        self.DNA_branch = EnformerConvStack(
            input_hsize = 4,
            target_hsize =target_hsize_DNA, # 512 reasonable default? is Enformer size divided by 3 TODO
            n_blocks = n_blocks_DNA,  # reasonable default? equals to Enformer setup TODO
            kernel_size = DNA_kernel_size, # reasonable default? TODO
            latent_size = latent_vector_size,
            dropout=DNA_dropout,
            progressive_channel_widening=progressive_channel_widening_DNA,
            pooling_between_blocks=pooling_between_blocks_DNA, # reasonable default? TODO
        )
        
        self.save_hyperparameters()



    def forward(self, x_DNA_in):
        x_DNA = self.DNA_branch(self.embedding(torch.tensor(x_DNA_in, dtype=torch.int)).permute(0, 2, 1))
        return x_DNA
        
    
    def configure_optimizers(self):
        optimizer = optim.Adam([
            {"params":self.DNA_branch.parameters(), "lr":self.learning_rate_DNA_branch}
            ])
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        x_DNA = train_batch["1/DNA_regions"]
        y = train_batch["central"]

        y_hat = self(x_DNA)
        loss = self.loss_function(y_hat.squeeze(), y.float().squeeze())


        self.log('train_loss', loss, prog_bar=True)
        return loss
    

    def validation_step(self, train_batch, batch_idx, dataloader_idx=0):
        x_DNA = train_batch["1/DNA_regions"]
        y = train_batch["central"]
        y_hat = self(x_DNA)
        
        if dataloader_idx == 0:
            loss = self.loss_function(y_hat.squeeze(), y.float().squeeze())
            self.log('val_loss', loss, prog_bar=True, add_dataloader_idx=False)
            AUROC = binary_auroc(y_hat.T.squeeze(), y.T.squeeze())
            self.log("AUROC", AUROC, add_dataloader_idx=False)
            Accuracy = binary_accuracy(y_hat.T.squeeze(), y.T.squeeze())
            self.log("Accuracy", Accuracy, add_dataloader_idx=False)
            return loss

        elif dataloader_idx == 1:
            self.log('val_loss_HQ', loss, prog_bar=True, add_dataloader_idx=False)
            AUROC = binary_auroc(y_hat.T.squeeze(), y.T.squeeze())
            self.log("AUROC_HQ", AUROC, add_dataloader_idx=False)
            Accuracy = binary_accuracy(y_hat.T.squeeze(), y.T.squeeze())
            self.log("Accuracy_HQ", Accuracy, add_dataloader_idx=False)
            

    def on_validation_epoch_end(self):
        pass


    def get_TF_latent_vector(self, TF_emb):
        return self.prot_branch(TF_emb.permute(0, 2, 1))
    

class TFmodel(pl.LightningModule):
    def __init__(
        self,

        # Model architecture
        target_hsize=64,
        n_blocks=2,
        DNA_kernel_size=9,
        progressive_channel_widening=True,
        pooling_between_blocks=True,
        DNA_dropout=0.25,

        # training
        learning_rate=1e-5,
    ):
        super(TFmodel, self).__init__()

        self.learning_rate = learning_rate
        self.loss_function = nn.BCEWithLogitsLoss()

        # nucleotide encodings
        nucleotide_weights = torch.FloatTensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0]]
            )
        self.embedding = nn.Embedding.from_pretrained(nucleotide_weights)
        self.best_metrics = {"AUROC": 0, "Accuracy": 0, "AUROC_HQ": 0, "Accuracy_HQ": 0}
        self.update_best_metrics_HQ = False
        
        # DNA branch
    
        self.DNA_branch = EnformerConvStack(
            input_hsize = 4,
            target_hsize =target_hsize, 
            n_blocks = n_blocks,
            kernel_size = DNA_kernel_size, 
            latent_size = 1, # because TF specific model
            dropout=DNA_dropout,
            progressive_channel_widening=progressive_channel_widening,
            pooling_between_blocks=pooling_between_blocks,
        )
        self.test_outputs_y = []
        self.test_outputs_y_hat = []
        self.HQ_test_outputs_y = []
        self.HQ_test_outputs_y_hat = []
        self.save_hyperparameters()



    def forward(self, x_DNA_in):
        x_DNA = self.DNA_branch(self.embedding(torch.tensor(x_DNA_in, dtype=torch.int)).permute(0, 2, 1))
        return x_DNA
        
    
    def configure_optimizers(self):
        optimizer = optim.Adam([
            {"params":self.DNA_branch.parameters(), "lr":self.learning_rate}
            ])
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        x_DNA = train_batch["1/DNA_regions"]
        y = train_batch["central"]

        y_hat = self(x_DNA)
        loss = self.loss_function(y_hat.squeeze(), y.float().squeeze())


        self.log('train_loss', loss, prog_bar=True)
        return loss
    

    def validation_step(self, train_batch, batch_idx, dataloader_idx=0):
        x_DNA = train_batch["1/DNA_regions"]
        y = train_batch["central"]
        y_hat = self(x_DNA)
        if dataloader_idx == 0:
            loss = self.loss_function(y_hat.squeeze(), y.float().squeeze())
            self.log('val_loss', loss, prog_bar=True, add_dataloader_idx=False)
            AUROC = binary_auroc(y_hat.T.squeeze(), y.T.squeeze())
            self.log("AUROC", AUROC, add_dataloader_idx=False)
            Accuracy = binary_accuracy(y_hat.T.squeeze(), y.T.squeeze())
            self.log("Accuracy", Accuracy, add_dataloader_idx=False)

            if AUROC > self.best_metrics["AUROC"]:
                self.best_metrics["AUROC"] = AUROC
                self.best_metrics["Accuracy"] = Accuracy
                self.update_best_metrics_HQ = True
                self.log("best_AUROC", AUROC, add_dataloader_idx=False)
                self.log("best_Accuracy", Accuracy, add_dataloader_idx=False)
            return loss

        elif dataloader_idx == 1:
            loss_HQ = self.loss_function(y_hat.squeeze(), y.float().squeeze())
            self.log('val_loss_HQ', loss_HQ, prog_bar=True, add_dataloader_idx=False)
            AUROC = binary_auroc(y_hat.T.squeeze(), y.T.squeeze())
            self.log("AUROC_HQ", AUROC, add_dataloader_idx=False)
            Accuracy = binary_accuracy(y_hat.squeeze(-1).T, y.T)
            self.log("Accuracy_HQ", Accuracy, add_dataloader_idx=False)
            if self.update_best_metrics_HQ:
                # always update the HQ metrics if the normal metrics are updated!
                self.best_metrics["AUROC_HQ"] = AUROC
                self.best_metrics["Accuracy_HQ"] = Accuracy
                self.log("best_AUROC_HQ", AUROC, add_dataloader_idx=False)
                self.log("best_Accuracy_HQ", Accuracy, add_dataloader_idx=False)
                self.update_best_metrics_HQ = False

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x_DNA = batch["1/DNA_regions"]
        y = batch["central"]
        y_hat = self(x_DNA)
        if dataloader_idx == 0:
            self.test_outputs_y.append(y)
            self.test_outputs_y_hat.append(y_hat)
        elif dataloader_idx == 1:
            self.HQ_test_outputs_y.append(y)
            self.HQ_test_outputs_y_hat.append(y_hat)

    def on_test_epoch_end(self):
        y_hat = torch.cat(self.test_outputs_y_hat, dim=0).squeeze()
        y_true = torch.cat(self.test_outputs_y, dim=0)
        AUROC = binary_auroc(y_hat, y_true)
        Accuracy = binary_accuracy(y_hat, y_true)
        self.log("test_AUROC", AUROC, prog_bar=True)
        self.log("test_Accuracy", Accuracy, prog_bar=True)

        y_hat_HQ = torch.cat(self.HQ_test_outputs_y_hat, dim=0).squeeze()
        y_true_HQ = torch.cat(self.HQ_test_outputs_y, dim=0)
        AUROC_HQ = binary_auroc(y_hat_HQ, y_true_HQ)
        Accuracy_HQ = binary_accuracy(y_hat_HQ, y_true_HQ)
        self.log("test_AUROC_HQ", AUROC_HQ, prog_bar=True)
        self.log("test_Accuracy_HQ", Accuracy_HQ, prog_bar=True)

            

class TFmodel_HQ(TFmodel):
    def validation_step(self, train_batch, batch_idx, dataloader_idx=0):
        x_DNA = train_batch["1/DNA_regions"]
        y = train_batch["central"]
        y_hat = self(x_DNA)
        if dataloader_idx == 0:
            loss = self.loss_function(y_hat.squeeze(), y.float().squeeze())
            self.log('val_loss', loss, prog_bar=True, add_dataloader_idx=False)
            AUROC = binary_auroc(y_hat.T.squeeze(), y.T.squeeze())
            self.log("AUROC", AUROC, add_dataloader_idx=False)
            Accuracy = binary_accuracy(y_hat.squeeze(-1).T, y.T)
            self.log("Accuracy", Accuracy, add_dataloader_idx=False)

            if AUROC > self.best_metrics["AUROC"]:
                self.best_metrics["AUROC"] = AUROC
                self.best_metrics["Accuracy"] = Accuracy
                self.update_best_metrics_HQ = True
                self.log("best_AUROC", AUROC, add_dataloader_idx=False)
                self.log("best_Accuracy", Accuracy, add_dataloader_idx=False)
            return loss
        
    def on_test_epoch_end(self):
        # We only need to modify this on epoch end, the test_step "elif dataloader_idx == 1:" will just never be used!
        y_hat = torch.cat(self.test_outputs_y_hat, dim=0).squeeze()
        y_true = torch.cat(self.test_outputs_y, dim=0)
        AUROC = binary_auroc(y_hat, y_true)
        Accuracy = binary_accuracy(y_hat, y_true)
        self.log("test_AUROC", AUROC, prog_bar=True)
        self.log("test_Accuracy", Accuracy, prog_bar=True)