import faiss 
import logging
import pickle
import glob
import os
import numpy as np

from accelerate.logging import get_logger

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self, 
        gpu_ids
    ):
        logger.info("Initializing retriever")
        self.doc_lookup = []
        self.gpu_ids = gpu_ids

    def _initialize_faiss_index(self, dim: int):
        self.index = None
        cpu_index = faiss.IndexFlatIP(dim)
        self.index = cpu_index

    def _move_index_to_gpu(self):
        ngpus = len(self.gpu_ids)
        gpu_ids = self.gpu_ids
        total = faiss.get_num_gpus()
        logger.info(f"Moving index to GPU(s) {ngpus} {total}")
        # print("total gpu", total)
        gpu_resources = []
        assert ngpus <= total, "Number of gpus for retrieval must less than total gpus"
        for i in range(ngpus):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpus):
            vdev.push_back(i)
            vres.push_back(gpu_resources[gpu_ids[i]])
        self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)
        self.index_on_gpu = True
    
    def init_index_and_add(self, root_dir, dataset_name):
        logger.info("Initializing Faiss index from pre-computed document embeddings")
        partitions = glob.glob(os.path.join(root_dir, f"{dataset_name}*"))
        for i, part in enumerate(partitions):
            logger.info(f"Load {part}")
            with open(part, "rb") as f:
                data = pickle.load(f)
            lookup_indices = data[0]
            encoded = data[1]
            # print(encoded.shape)
            # print(lookup_indices.shape)
            if i == 0:
                dim = encoded.shape[1]
                self._initialize_faiss_index(dim)
            self.index.add(encoded)
            self.doc_lookup.extend(lookup_indices)
            # break
        self._move_index_to_gpu()
    
    def search(self, query_ids, query_embs, top_k):
        return_dict = {}
        D, I = self.index.search(query_embs, top_k)
        original_indices = np.array(self.doc_lookup)[I]
        for qid, scores_per_q, doc_indices_per_q in zip(query_ids, D, original_indices):
            return_dict[qid] = {}
            for doc_index, score in zip(doc_indices_per_q, scores_per_q):
                doc_index = str(doc_index)
                return_dict[qid][doc_index] = float(score)

        return return_dict
    

if __name__ == "__main__":
    retrieve = Retriever()
    os.makedirs("save_embs", exist_ok=True)
    retrieve.init_index_and_add(root_dir="save_embs/gte-ml-base", dataset_name="dpr100")
