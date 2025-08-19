import logging
import os
import numpy as np
from pathlib import Path
from time import time
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import AutoTokenizer, AutoModel
from src.data_sampler import CorpusCollator, CorpusDatasetForEmbedding
import ir_datasets
import torch.nn.functional as F

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = get_logger(__name__, log_level="INFO")

def main():
    ir_dataset = ir_datasets.load(f"dpr-w100/natural-questions/train")
    docstore = ir_dataset.docs_store()
    doc_c = ir_dataset.docs_count()
    accelerator = Accelerator()
    
    model_name_or_path = 'hf/gte-multilingual-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    corpus_dataset = CorpusDatasetForEmbedding(docstore=docstore, n_docs=doc_c)
    logger.info(f"Sample a passage: {corpus_dataset.__getitem__(1)}")
    corpus_collator = CorpusCollator(tokenizer=tokenizer)
    corpus_dataloader = DataLoader(
        corpus_dataset, shuffle=False, collate_fn=corpus_collator, batch_size=512
    )

    model, corpus_dataloader = accelerator.prepare(
        model, corpus_dataloader
    )

    os.makedirs("emb_corpus/gte-ml-base", exist_ok=True)
    samples_seen = 0
    progress_bar_eval = tqdm(range(len(corpus_dataloader)), disable=not accelerator.is_local_main_process)
    dimension=768
    all_ids = []
    all_embs = []
    for step, batch in enumerate(corpus_dataloader):
        progress_bar_eval.update(1)
        corpus_ids = batch.pop("corpus_ids")
        with torch.no_grad():
            outputs = model(**batch["passage"])
        embeddings = outputs.last_hidden_state[:, 0][:dimension]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        p_reps, corpus_ids = accelerator.gather((embeddings, corpus_ids))
        
        if accelerator.num_processes > 1:
            if step == len(corpus_dataloader) - 1:
                p_reps = p_reps[:len(corpus_dataloader.dataset) - samples_seen]
                corpus_ids = corpus_ids[:len(corpus_dataloader.dataset) - samples_seen]
            else:
                samples_seen += corpus_ids.shape[0]

        torch.cuda.empty_cache()
        p_reps = p_reps.cpu().numpy()
        corpus_ids = corpus_ids.cpu().numpy()
        all_ids.append(corpus_ids)
        all_embs.append(p_reps)
        if (step + 1) % 2000 == 0:
            if accelerator.is_main_process:
                all_ids = np.concatenate(all_ids, 0)
                all_embs = np.concatenate(all_embs, 0)
                with open(f"emb_corpus/gte-ml-base/dpr100_{step}", mode="wb") as f:
                    pickle.dump((all_ids, all_embs), f)        
                all_ids = []
                all_embs = []
        
    if accelerator.is_main_process:
        all_ids = np.concatenate(all_ids, 0)
        all_embs = np.concatenate(all_embs, 0)
        with open(f"emb_corpus/gte-ml-base/dpr100_{step}", mode="wb") as f:
            pickle.dump((all_ids, all_embs), f)



if __name__ == "__main__":
    main()
