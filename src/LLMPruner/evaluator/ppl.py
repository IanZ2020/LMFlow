import torch
import numpy as np
from tqdm import tqdm
from lmflow.datasets.dataset import Dataset
import torch.distributed as dist

from LLMPruner.datasets.ppl_dataset import get_loaders

def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric

@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits
    
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    #print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


def evaluate_ppl(model, tokenizer, dataset: Dataset, batch_size=8, block_size = 1024, verbose=True):
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        data_dict = dataset.to_dict()
        if data_dict['type'] == 'text2text':
            raise NotImplementedError("ppl evaluation is currently not supported for text2text dataset, please use text_only dataset.")
        texts = [ instance["text"] for instance in data_dict["instances"] ]
        encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
        # Define some constant
        max_length = block_size
        
        if verbose:
            print(f"The maximum sequence length : {max_length}")
        encode_batch_num = encodings.input_ids.size(0)
        seq_len = encodings.input_ids.size(1)

        nlls = []
        len_per_device = seq_len // world_size
        current_batch = encodings.input_ids[:, local_rank*len_per_device:(local_rank+1)*len_per_device]

        num_of_example = len_per_device // block_size
        num_of_batch = num_of_example // batch_size * encode_batch_num
        current_batch = current_batch[:,0: num_of_batch*batch_size*block_size].view(num_of_batch, batch_size, block_size)

        count = 0
        for id in range(0, num_of_batch):
            input_ids = current_batch[id].to(device=local_rank)
            target_ids = input_ids.clone()
            trg_len = block_size
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            count += 1

            if verbose:
                print(f"rank{local_rank}, Evaluating PPL: {count} / {num_of_batch} Complete, current ppl : {torch.exp(torch.stack(nlls).mean())}")
        
        all_process = torch.stack(nlls).mean()
        dist.all_reduce(all_process, dist.ReduceOp.SUM, async_op=False)
        result = all_process / world_size
        ppl = torch.exp(result)
        return ppl