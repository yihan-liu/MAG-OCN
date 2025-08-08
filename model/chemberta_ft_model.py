from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class MAGChemBERTa(nn.Module):
    """ChemBERTa encoder *augmented* with learnable 3D coordinate embeddings for magnetic moment prediction.

    The pretrained chemical language model provides rich contextual features for
    atom *identities* (via the SMILES string). We project each atom's (x, y, z)
    coordinates into the same hidden dimension and **add** that vector to the
    corresponding token hidden state before the final encoder layer.

    Assumptions
    -----------
      - The SMILES string is generated **with explicit hydrogens** and with an atom
        ordering that matches the coordinate array coming from
        :pyfunc:`OCNMoleculeDataset`: ``features[:, :n_atom_types]``.
      - A mapping `token2atom` (list[int]) tells which token index corresponds to
        which atom in the 3D coordinate array.
    """

    def __init__(
        self,
        pretrained_name: str = 'seyonec/ChemBERTa-zinc-base-v1',
        coord_dim: int = 3,
        lora_r: int = 8,
    ):
        super().__init__()
        self.chembert = AutoModel.from_pretrained(pretrained_name)
        self.hidden = self.chembert.config.hidden_size

        # simple linear projection of (x,y,z) to hidden dimension
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )

        # PEFT with LoRA
        try:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=lora_r,
                target_modules=['query', 'value'],  # ChemBERTa/RoBERTa uses 'query', 'value' not 'q_proj', 'v_proj'
                bias='none',
                task_type='FEATURE_EXTRACTION',  # Changed from SEQ_CLS to FEATURE_EXTRACTION
            )
            self.chembert = get_peft_model(self.chembert, lora_cfg)
        except ImportError:
            print("PEFT not installed; continuing without LoRA.")
        except ValueError as e:
            print(f"LoRA configuration error: {e}")
            print("Continuing without LoRA.")

        # heads
        self.mm_head = nn.Linear(self.hidden, 1)  # magnetic moment regression

    @staticmethod
    def add_coord_embeds(last_hidden: torch.Tensor, coords: torch.Tensor, token2atom: list[int], coord_mlp: nn.Module) -> torch.Tensor:
        """Add coordinate embeddings to the last hidden layer **in-place**.

        Parameters
        ----------
        last_hidden : torch.Tensor [L, H]
            Output hidden states of ChemBERTa for one molecule (L tokens).
        coords : torch.Tensor [N, 3]
            Normalised (x, y, z) of each atom.
        token2atom : list[int]
            ``token2atom[i] = j`` means token *i* corresponds to atom *j*.
            Should only map chemical tokens (skip [CLS], [SEP], etc.).
        coord_mlp : nn.Module
            Projection network from 3D → hidden_size.
        """
        device = last_hidden.device
        coord_emb = coord_mlp(coords.to(device))  # [N, H]
        
        # Skip first ([CLS]) and last ([SEP]) tokens - only add coords to chemical tokens
        for tok_idx, atom_idx in enumerate(token2atom):
            if tok_idx > 0 and tok_idx < len(token2atom) - 1:  # Skip special tokens
                if atom_idx < len(coord_emb):  # Bounds check
                    last_hidden[tok_idx] += coord_emb[atom_idx]

    @staticmethod
    def get_tokenizer(pretrained_name: str = 'seyonec/ChemBERTa-zinc-base-v1'):
        """Get the tokenizer for the ChemBERTa model."""
        return AutoTokenizer.from_pretrained(pretrained_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        token2atom: Optional[list] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning magnetic moment predictions.
        
        Parameters
        ----------
        input_ids : torch.Tensor [B, L]
            Tokenized SMILES sequences.
        attention_mask : torch.Tensor [B, L], optional
            Attention mask for the sequences.
        coords : torch.Tensor [B, N, 3], optional
            3D coordinates for each atom.
        token2atom : list[list[int]], optional
            Token-to-atom mapping for each sample in the batch.
        mask : torch.Tensor [B, N], optional
            Mask indicating valid atoms.
            
        Returns
        -------
        torch.Tensor [B, N]
            Predicted magnetic moments for each atom.
        """
        out = self.chembert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]  # [B, L, H]

        # add coord embeddings per sample
        if coords is not None and token2atom is not None:
            for b in range(hidden.size(0)):
                self.add_coord_embeds(
                    hidden[b],            # [L, H] for sample b
                    coords[b],            # [N, 3] for sample b
                    token2atom[b],        # <- pass the *per-sample* mapping
                    self.coord_mlp,
                )

        # per-atom magnetic moment regression
        B, L, H = hidden.size()
        device = hidden.device
        max_atoms = coords.shape[1] if coords is not None else L
        atom_h = torch.zeros(B, max_atoms, H, device=device)
        
        if token2atom is not None:
            for b, mapping in enumerate(token2atom):
                for tok_idx, atom_idx in enumerate(mapping):
                    # Skip special tokens ([CLS] at 0, [SEP] at end) when mapping back to atoms
                    if tok_idx > 0 and tok_idx < len(mapping) - 1:  # Only chemical tokens
                        if tok_idx < L and atom_idx < max_atoms:
                            atom_h[b, atom_idx] = hidden[b, tok_idx]
        else:
            # fallback: use token embeddings directly (skip first/last for special tokens)
            atom_h = hidden[:, 1:-1, :]  # Skip [CLS] and [SEP]

        mm_pred = self.mm_head(atom_h).squeeze(-1)  # [B, N]
        return mm_pred
