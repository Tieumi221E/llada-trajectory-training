import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any
from .masking import apply_diffusion_mask

class SimpleTextDataset(Dataset):
    """
    A simple dataset that takes a list of pre-tokenized ID lists or strings.
    """
    def __init__(self, tokenized_texts: List[List[int]]):
        self.data = tokenized_texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DiffusionCollator:
    def __init__(
        self,
        tokenizer: Any,
        mask_token_id: int,
        max_length: int,
        eps: float = 1e-3,
        mode: str = "pretrain", # "pretrain" or "sft"
    ):
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.max_length = max_length
        self.eps = eps
        self.mode = mode

    def __call__(self, batch: List[List[int]]) -> Dict[str, torch.Tensor]:
        # 1. Padding
        max_batch_len = min(self.max_length, max(len(x) for x in batch))
        
        input_ids_list = []
        attention_mask_list = []
        valid_mask_list = [] # Mask for tokens that CAN be masked
        
        for tokens in batch:
            tokens = tokens[:self.max_length]
            length = len(tokens)
            pad_len = max_batch_len - length
            
            # Pad with tokenizer pad_token_id
            padded_ids = tokens + [self.tokenizer.pad_token_id] * pad_len
            input_ids_list.append(padded_ids)
            
            # Attention mask (1 for real, 0 for pad)
            att_mask = [1] * length + [0] * pad_len
            attention_mask_list.append(att_mask)
            
            if self.mode == "pretrain":
                # In pretrain, all real tokens can be masked
                valid_mask_list.append(att_mask)
            else:
                # In SFT, we assume the input is [Prompt, Response]
                # And we only want to mask the Response part.
                # NOTE: For this to work generically, the user should provide 
                # (prompt_ids, response_ids) or we need a marker.
                # For this generic collator, let's assume the whole sequence 
                # is valid unless the user uses a more specific dataset.
                valid_mask_list.append(att_mask)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
        valid_mask = torch.tensor(valid_mask_list, dtype=torch.long)
        
        # 2. Apply Masking
        noisy_input, target_ids, _, p_scalar = apply_diffusion_mask(
            clean_input=input_ids,
            attention_mask=attention_mask,
            mask_token_id=self.mask_token_id,
            valid_mask=valid_mask,
            eps=self.eps
        )
        
        return {
            "input_ids": noisy_input,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "p_mask_scalar": p_scalar
        }
