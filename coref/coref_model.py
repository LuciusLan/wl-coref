""" see __init__.py """

from datetime import datetime
import os
import pickle
import random
import re
import itertools
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np      # type: ignore
import jsonlines        # type: ignore
import toml
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm   # type: ignore
import transformers     # type: ignore
import json

from coref import bert, conll, utils
from coref.anaphoricity_scorer import AnaphoricityScorer, AnaphoricityScorerChunk
from coref.cluster_checker import ClusterChecker
from coref.config import Config
from coref.const import CorefResult, Doc, Span
from coref.loss import CorefLoss
from coref.pairwise_encoder import PairwiseEncoder, PairwiseEncoderChunk
from coref.chunk_encoder import ChunkEncoder
from coref.rough_scorer import RoughScorer, RoughScorerChunk
from coref.span_predictor import SpanPredictor, SpanPredictorChunk
from coref.tokenizer_customization import TOKENIZER_FILTERS, TOKENIZER_MAPS
from coref.utils import GraphNode, non_max_sup
from coref.word_encoder import WordEncoder


class CorefModel:  # pylint: disable=too-many-instance-attributes
    """Combines all coref modules together to find coreferent spans.

    Attributes:
        config (coref.config.Config): the model's configuration,
            see config.toml for the details
        epochs_trained (int): number of epochs the model has been trained for
        trainable (Dict[str, torch.nn.Module]): trainable submodules with their
            names used as keys
        training (bool): used to toggle train/eval modes

    Submodules (in the order of their usage in the pipeline):
        tokenizer (transformers.AutoTokenizer)
        bert (transformers.AutoModel)
        we (WordEncoder)
        rough_scorer (RoughScorer)
        pw (PairwiseEncoder)
        a_scorer (AnaphoricityScorer)
        sp (SpanPredictor)
    """
    def __init__(self,
                 config_path: str,
                 section: str,
                 epochs_trained: int = 0):
        """
        A newly created model is set to evaluation mode.

        Args:
            config_path (str): the path to the toml file with the configuration
            section (str): the selected section of the config file
            epochs_trained (int): the number of epochs finished
                (useful for warm start)
        """
        self.config = CorefModel._load_config(config_path, section)
        self.epochs_trained = epochs_trained
        self._docs: Dict[str, List[Doc]] = {}
        self._build_model()
        self._build_optimizers()
        self._set_training(False)
        self._coref_criterion = CorefLoss(self.config.bce_loss_weight)
        self._span_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        #self._tokenize_docs('data/english_test_head.jsonlines', 'test')

    @property
    def training(self) -> bool:
        """ Represents whether the model is in the training mode """
        return self._training

    @training.setter
    def training(self, new_value: bool):
        if self._training is new_value:
            return
        self._set_training(new_value)

    # ========================================================== Public methods


    def case_print(self, doc, clusters):
        for i, clus in enumerate(clusters):
            print(f"Cluster {i}")
            for span in clus:
                try:
                    print(f"Context: {doc['cased_words'][span[0]-5:span[1]+5]}")
                    print(f"Entity: {doc['cased_words'][span[0]:span[1]]}\n")
                except IndexError:
                    print(f"Entity: {doc['cased_words'][span[0]:span[1]]}\n")

    def case_print_cl(self, doc, clusters):
        new_clusters = []
        for i, clus in enumerate(clusters):
            new_clusters.append([])
            
            prev = 999
            temp_cont = []
            for i, chunk in enumerate(clus):
                if chunk == prev+1:
                    temp_cont.append(chunk)
                else:
                    if temp_cont != []:
                        new_clusters[-1].append(temp_cont)
                    temp_cont = [chunk]
                if i == len(clus) -1:
                    new_clusters[-1].append(temp_cont)
                prev = chunk
        for i, clus in enumerate(new_clusters):
            print(f"Cluster {i}")
            for chunk in clus:
                if len(chunk) > 1:
                    try:
                        span = [doc['splitted_chunk'][chunk[0]][0], doc['splitted_chunk'][chunk[-1]][1]]
                        print(f"Context: {doc['cased_words'][span[0]-5:span[1]+5]}")
                        print(f"Entity: {doc['cased_words'][span[0]:span[1]]}\n")
                    except IndexError:
                        print(f"Entity: {doc['cased_words'][span[0]:span[1]]}\n")
                else:
                    chunk = chunk[0]
                    try:
                        span = doc['splitted_chunk'][chunk]
                        print(f"Context: {doc['cased_words'][span[0]-5:span[1]+5]}")
                        print(f"Entity: {doc['cased_words'][span[0]:span[1]]}\n")
                    except IndexError:
                        print(f"Entity: {doc['cased_words'][span[0]:span[1]]}\n")

    @torch.no_grad()
    def evaluate(self,
                 data_split: str = "dev",
                 word_level_conll: bool = False,
                 sp_model=None
                 ) -> Tuple[float, Tuple[float, float, float]]:
        """ Evaluates the modes on the data split provided.

        Args:
            data_split (str): one of 'dev'/'test'/'train'
            word_level_conll (bool): if True, outputs conll files on word-level

        Returns:
            mean loss
            span-level LEA: f1, precision, recal
        """
        self.training = False
        w_checker = ClusterChecker()
        s_checker = ClusterChecker()
        docs = self._get_docs(self.config.__dict__[f"{data_split}_data"])
        running_loss = 0.0
        s_correct = 0
        s_total = -1

        with conll.open_(self.config, self.epochs_trained, data_split) \
                as (gold_f, pred_f):
            pbar = tqdm(docs, unit="docs", ncols=0)
            for doc in pbar:
                res = self.run(doc)
                #sp_res = sp_model.run(doc, True, res.word_clusters)

                #running_loss += self._coref_criterion(res.coref_scores, res.coref_y).item()
                running_loss += self._coref_criterion(res.coref_scores_chunk, res.coref_y).item()
                
                if res.span_y:
                    """pred_starts = sp_res.span_scores[:, :, 0].argmax(dim=1)
                    pred_ends = sp_res.span_scores[:, :, 1].argmax(dim=1)
                    s_correct += ((sp_res.span_y[0] == pred_starts) * (sp_res.span_y[1] == pred_ends)).sum().item()"""
                    pred_starts = res.span_scores[:, :, 0].argmax(dim=1)
                    pred_ends = res.span_scores[:, :, 1].argmax(dim=1)
                    s_correct += ((res.span_y[0] == pred_starts) * (res.span_y[1] == pred_ends)).sum().item()
                    s_total += len(pred_starts)

                if word_level_conll:
                    conll.write_conll(doc,
                                      [[(i, i + 1) for i in cluster]
                                       for cluster in doc["word_clusters"]],
                                      gold_f)
                    conll.write_conll(doc,
                                      [[(i, i + 1) for i in cluster]
                                       for cluster in res.word_clusters],
                                      pred_f)
                else:
                    conll.write_conll(doc, doc["span_clusters"], gold_f)
                    conll.write_conll(doc, res.span_clusters, pred_f)

                #w_checker.add_predictions(doc["word_clusters"], res.word_clusters)
                
                chunk_clus_target = [[] for i in range(len(doc['cluster_ids_chunk']))]
                temp = torch.tensor(doc['cluster_ids_chunk'].copy()).nonzero().tolist()
                for e in temp:
                    chunk_clus_target[e[0]].append(e[1])

                #self.case_print_cl(doc, chunk_clus_target)
                
                w_checker.add_predictions(chunk_clus_target, res.word_clusters)
                w_lea = w_checker.total_lea

                s_checker.add_predictions(doc["span_clusters"], res.span_clusters)
                s_lea = s_checker.total_lea

                del res

                pbar.set_description(
                    f"{data_split}:"
                    f" | WL: "
                    f" loss: {running_loss / (pbar.n + 1):<.5f},"
                    f" f1: {w_lea[0]:.5f},"
                    f" p: {w_lea[1]:.5f},"
                    f" r: {w_lea[2]:<.5f}"
                    f" | SL: "
                    f" sa: {s_correct / s_total:<.5f},"
                    f" f1: {s_lea[0]:.5f},"
                    f" p: {s_lea[1]:.5f},"
                    f" r: {s_lea[2]:<.5f}"
                )
                
                torch.cuda.empty_cache()
            self.logger.info(
                f"{data_split}:"
                f" | WL: "
                f" loss: {running_loss / (pbar.n + 1):<.5f},"
                f" f1: {w_lea[0]:.5f},"
                f" p: {w_lea[1]:.5f},"
                f" r: {w_lea[2]:<.5f}"
                f" | SL: "
                f" sa: {s_correct / s_total:<.5f},"
                f" f1: {s_lea[0]:.5f},"
                f" p: {s_lea[1]:.5f},"
                f" r: {s_lea[2]:<.5f}"
            )

        return (running_loss / len(docs), *s_checker.total_lea)

    def run(self,  # pylint: disable=too-many-locals
            doc: Doc,
            ) -> CorefResult:
        """
        This is a massive method, but it made sense to me to not split it into
        several ones to let one see the data flow.

        Args:
            doc (Doc): a dictionary with the document data.

        Returns:
            CorefResult (see const.py)
        """
        # Encode words with bert
        # words           [n_words, span_emb]
        # cluster_ids     [n_words]
        words, cluster_ids = self.we(doc, self._bertify(doc))
        
        subwords = self._bertify(doc)
        #words, _ = self.we(doc, self._bertify(doc))
        # Obtain bilinear scores and leave only top-k antecedents for each word
        # top_rough_scores  [n_words, n_ants]
        # top_indices       [n_words, n_ants]

        chunk_pos = doc["splitted_chunk"].copy()
        start_pos = torch.tensor(chunk_pos, dtype=torch.long, device=self.config.device)
                
        mask_range = torch.arange(len(doc['cased_words']), device=self.config.device)
        mask_starts = start_pos[:, 0].unsqueeze(1)
        mask_ends = start_pos[:, 1].unsqueeze(1)
        slice_masks = ((mask_range >= mask_starts) & (mask_range < mask_ends)).float()
        slice_sum = slice_masks.sum(1)


        #chunk_feature = self.chunk_aux(doc, slice_sum, start_pos[:,0])

        #chunks = torch.cat([words[start_pos[:,0]], words[start_pos[:,1]-1], torch.div(slice_masks.mm(words).transpose(0,1), slice_sum).transpose(0,1)],dim=1)
        chunks = words[start_pos[:,0]]
        """chunks = []
        for i, (s, e) in enumerate(doc["splitted_chunk"]):
            chunks.append(torch.cat([words[s], words[e-1], torch.einsum('ij->i',words[s:e].transpose(0,1))/(e-s)],0))"""


        #for i, (s, e) in enumerate(doc["extra_chunks"]):
        #    chunks.append(words[s].tile(3))
        #chunks = torch.stack(chunks)
        #chunk_pos = doc["chunk_list"].copy()
        #chunk_pos.extend(doc['extra_chunks'])
        start_pos = start_pos[:,0]

        """single_chunks = []
        single_words = []
        nonsingle = []
        
        for i, (s, e) in enumerate(doc["single_chunk_pos"]):
            single_chunks.append(words[s].tile(3))
            single_words.append(words[s])
        for i, (s, e) in enumerate(doc["nonsingle_chunk_pos"]):
            nonsingle.append(torch.cat([words[s], words[e-1], torch.einsum('ij->i',words[s:e].transpose(0,1))/(e-s)]))

        single_chunks = torch.stack(single_chunks)
        single_words = torch.stack(single_words)
        nonsingle = torch.stack(nonsingle)
        chunks = torch.cat([single_chunks, nonsingle], 0)
        chunk_pos = doc["single_chunk_pos"].copy()
        chunk_pos.extend(doc["nonsingle_chunk_pos"])
        chunk_pos = torch.tensor(chunk_pos, dtype=torch.long, device=self.config.device)
        start_pos = chunk_pos[:, 0]"""


        cluster_ids_chunk = torch.LongTensor(doc['cluster_ids_chunk']).to(self.config.device)
        if cluster_ids_chunk.size(0) == 0:
            cluster_ids_chunk = torch.zeros(chunks.size(0), dtype=torch.long, device=self.config.device).unsqueeze(0)

        #top_rough_scores, top_indices = self.rough_scorer(single_chunks)
        top_rough_scores = self.rough_scorer_chunk(chunks, chunks, False)
        top_rough_scores, top_indices = self.rough_scorer_chunk._prune(top_rough_scores)
        
        """one2one_scores = self.rough_scorer(single_words)
        n2one_scores = self.rough_scorer_chunk(nonsingle, single_chunks, is_one2n=True)
        n2n_scores = self.rough_scorer_chunk(nonsingle, nonsingle, is_one2n=False)
        one2n_scores = torch.log(torch.zeros_like(n2one_scores.T))
        top_rough_scores = torch.cat((torch.cat((one2one_scores, n2one_scores), 0), torch.cat((one2n_scores, n2n_scores), 0)), 1)
        top_rough_scores, top_indices = self.rough_scorer._prune(top_rough_scores)"""
        # Get pairwise features [n_words, n_ants, n_pw_features]
        #pw_chunks = self.pw_chunk(top_indices, doc, chunk_pos, start_pos)
        pw_chunks = self.pw_chunk(top_indices, doc, chunk_pos, start_pos)

        batch_size = self.config.a_scoring_batch_size
        #a_scores_lst: List[torch.Tensor] = []
        a_scores_chunk_lst = []

        """for i in range(0, len(single_chunks), batch_size):
            pw_batch = pw[i:i + batch_size]
            words_batch = single_chunks[i:i + batch_size]
            top_indices_batch = top_indices[i:i + batch_size]
            top_rough_scores_batch = top_rough_scores[i:i + batch_size]

            # a_scores_batch    [batch_size, n_ants]
            a_scores_batch = self.a_scorer(
                all_mentions=single_chunks, mentions_batch=words_batch,
                pw_batch=pw_batch, top_indices_batch=top_indices_batch,
                top_rough_scores_batch=top_rough_scores_batch
            )
            a_scores_lst.append(a_scores_batch)"""
        
        res = CorefResult()
        if len(chunks) > 1500:
            res.long_doc_flag = True

        # coref_scores  [n_spans, n_ants]
        for i in range(0, len(chunks), batch_size):
            pw_batch = pw_chunks[i:i + batch_size]
            words_batch = chunks[i:i + batch_size]
            top_indices_batch = top_indices[i:i + batch_size]
            top_rough_scores_batch = top_rough_scores[i:i + batch_size]

            # a_scores_batch    [batch_size, n_ants]
            a_scores_batch = self.a_scorer_chunk(
                all_mentions=chunks, mentions_batch=words_batch,
                pw_batch=pw_batch, top_indices_batch=top_indices_batch,
                top_rough_scores_batch=top_rough_scores_batch
            )
            a_scores_chunk_lst.append(a_scores_batch)
            
        #res.coref_scores = torch.cat(a_scores_lst, dim=0)

        res.coref_scores_chunk = torch.cat(a_scores_chunk_lst, dim=0)

        #res.coref_y = self._get_ground_truth(
        #    cluster_ids, top_indices, (top_rough_scores > float("-inf")))
                
        # TODO: clustering for chunks
        #res.word_clusters = self._clusterize(doc, res.coref_scores,
        #                                     top_indices)
        
        res.coref_y = self._get_ground_truth_chunk(
            cluster_ids_chunk, top_indices, chunk_pos, (top_rough_scores > float("-inf")))
        res.word_clusters = self._clusterize_chunk(doc, res.coref_scores_chunk,
                                             top_indices)
        with torch.cuda.amp.autocast(enabled=False):
            #res.span_scores, res.span_y = self.sp.get_training_data(doc, words, False)
        #res.span_scores, res.span_y = self.sp.get_training_data(doc, words, True)
            res.span_scores, res.span_y = self.sp_chunk.get_training_data(doc, chunks, True, start_pos)

        """if not self.training:
            temp_word_clus = []
            for cluster in res.word_clusters:
                temp_word_clus.append([])
                for chunk in cluster:
                    temp_word_clus[-1].extend([wp for wp in range(chunk_pos[chunk][0], chunk_pos[chunk][1])])
            temp_clus = self.sp.predict(doc, words, temp_word_clus)
            res.span_clusters = [non_max_sup(cluster) for cluster in temp_clus]"""
        if not self.training:
            res.span_clusters = self.sp_chunk.predict(doc, chunks, res.word_clusters, start_pos, chunk_pos)
        #if not self.training:
        #    res.span_clusters = self.sp_chunk.predict_direct(res.word_clusters, chunk_pos)

        return res
    
    @staticmethod
    def _get_ground_truth(cluster_ids: torch.Tensor,
                          top_indices: torch.Tensor,
                          valid_pair_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cluster_ids: tensor of shape [n_words], containing cluster indices
                for each word. Non-gold words have cluster id of zero.
            top_indices: tensor of shape [n_words, n_ants],
                indices of antecedents of each word
            valid_pair_map: boolean tensor of shape [n_words, n_ants],
                whether for pair at [i, j] (i-th word and j-th word)
                j < i is True

        Returns:
            tensor of shape [n_words, n_ants + 1] (dummy added),
                containing 1 at position [i, j] if i-th and j-th words corefer.
        """
        y = cluster_ids[top_indices] * valid_pair_map  # [n_words, n_ants]
        y[y == 0] = -1                                 # -1 for non-gold words
        y = utils.add_dummy(y)                         # [n_words, n_cands + 1]
        y = (y == cluster_ids.unsqueeze(1))            # True if coreferent
        # For all rows with no gold antecedents setting dummy to True
        y[y.sum(dim=1) == 0, 0] = True
        return y.to(torch.float)

    @staticmethod
    def _get_ground_truth_chunk(cluster_ids: torch.Tensor,
                          top_indices: torch.Tensor,
                          chunk_pos,
                          valid_pair_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cluster_ids: tensor of shape [n_words], containing cluster indices
                for each word. Non-gold words have cluster id of zero.
            top_indices: tensor of shape [n_words, n_ants],
                indices of antecedents of each word
            valid_pair_map: boolean tensor of shape [n_words, n_ants],
                whether for pair at [i, j] (i-th word and j-th word)
                j < i is True

        Returns:
            tensor of shape [n_words, n_ants + 1] (dummy added),
                containing 1 at position [i, j] if i-th and j-th words corefer.
        """
        y = torch.zeros_like(top_indices)
        for row in cluster_ids:
            y += row[top_indices] * valid_pair_map
        #y = cluster_ids[top_indices] * valid_pair_map  # [n_words, n_ants]
        y[y == 0] = -1                                 # -1 for non-gold words
        y = utils.add_dummy(y)                         # [n_words, n_cands + 1]
        y = (y == cluster_ids.sum(0).unsqueeze(1))            # True if coreferent
        # For all rows with no gold antecedents setting dummy to True
        y[y.sum(dim=1) == 0, 0] = True
        return y.to(torch.float)

    def save_weights(self):
        """ Saves trainable models as state dicts. """
        to_save: List[Tuple[str, Any]] = \
            [(key, value) for key, value in self.trainable.items()
             if self.config.bert_finetune or key != "bert"]
        to_save.extend(self.optimizers.items())
        to_save.extend(self.schedulers.items())

        time = datetime.strftime(datetime.now(), "%Y.%m.%d_%H.%M")
        """path = os.path.join(self.config.data_dir,
                            f"chunk_{self.config.section}"
                            f"_(e{self.epochs_trained}_{time}).pt")"""
        path = os.path.join(self.config.data_dir,
                            f"chunk_{self.config.section}_best.pt")
        savedict = {name: module.state_dict() for name, module in to_save}
        savedict["epochs_trained"] = self.epochs_trained  # type: ignore
        torch.save(savedict, path)

    def load_weights(self,
                     path: Optional[str] = None,
                     ignore: Optional[Set[str]] = None,
                     map_location: Optional[str] = None,
                     noexception: bool = False) -> None:
        """
        Loads pretrained weights of modules saved in a file located at path.
        If path is None, the last saved model with current configuration
        in data_dir is loaded.
        Assumes files are named like {configuration}_(e{epoch}_{time})*.pt.
        """
        if path is None:
            #pattern = rf"chunk_{self.config.section}_\(e(\d+)_[^()]*\).*\.pt"
            pattern = rf"chunk_{self.config.section}_best.pt"
            files = []
            for f in os.listdir(self.config.data_dir):
                match_obj = re.match(pattern, f)
                if match_obj:
                    #files.append((int(match_obj.group(0)), f))
                    path = match_obj.group(0)
            #if not files:
            if not path:
                if noexception:
                    self.logger.info("No weights have been loaded", flush=True)
                    return
                raise OSError(f"No weights found in {self.config.data_dir}!")
            #_, path = sorted(files)[-1]
            path = os.path.join(self.config.data_dir, path)

        if map_location is None:
            map_location = self.config.device
        self.logger.info(f"Loading from {path}...")
        state_dicts = torch.load(path, map_location=map_location)
        self.epochs_trained = state_dicts.pop("epochs_trained", 0)
        for key, state_dict in state_dicts.items():
            if not ignore or key not in ignore:
                if key.endswith("_optimizer"):
                    self.optimizers[key].load_state_dict(state_dict)
                elif key.endswith("_scheduler"):
                    self.schedulers[key].load_state_dict(state_dict)
                else:
                    self.trainable[key].load_state_dict(state_dict)
                self.logger.info(f"Loaded {key}")

    def train(self):
        """
        Trains all the trainable blocks in the model using the config provided.
        """
        docs = list(self._get_docs(self.config.train_data))
        docs_ids = list(range(len(docs)))
        #avg_spans = sum(len(doc["head2span"]) for doc in docs) / len(docs)
        avg_spans = sum(len(doc["chunk_head"]) for doc in docs) / len(docs)

        for epoch in range(self.epochs_trained, self.config.train_epochs):
            self.training = True
            running_c_loss = 0.0
            running_s_loss = 0.0
            random.shuffle(docs_ids)
            pbar = tqdm(docs_ids, unit="docs", ncols=0, total=len(docs))
            for step, doc_id in enumerate(pbar):
                doc = docs[doc_id]
                if step == 417:
                    print()

                for optim in self.optimizers.values():
                    optim.zero_grad()
                with torch.cuda.amp.autocast():
                    res = self.run(doc)
                    long_doc_flag = res.long_doc_flag

                    #c_loss = self._coref_criterion(res.coref_scores, res.coref_y)
                    c_loss = self._coref_criterion(res.coref_scores_chunk, res.coref_y)
                    if res.span_y:
                        s_loss = (self._span_criterion(res.span_scores[:, :, 0], res.span_y[0])
                                + self._span_criterion(res.span_scores[:, :, 1], res.span_y[1])) / avg_spans / 2
                    else:
                        s_loss = torch.zeros_like(c_loss)

                    if s_loss == math.inf:
                        print()
                    del res
                    if long_doc_flag:
                        torch.cuda.empty_cache()

                    (c_loss + s_loss).backward()
                    running_c_loss += c_loss.item()
                    running_s_loss += s_loss.item()

                del c_loss, s_loss
                for module in self.trainable.values():
                    clip_grad_norm_(module.parameters(), 5.0)
                for optim in self.optimizers.values():
                    optim.step()
                for scheduler in self.schedulers.values():
                    scheduler.step()

                pbar.set_description(
                    f"Epoch {epoch + 1}:"
                    f" {doc['document_id']:26}"
                    f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                    f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
                )
                torch.cuda.empty_cache()
            
            self.logger.info(
                    f"Epoch {epoch + 1}:"
                f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
            )
            self.epochs_trained += 1
            self.save_weights()
            self.evaluate()

    # ========================================================= Private methods

    def _bertify(self, doc: Doc) -> torch.Tensor:
        subwords_batches = bert.get_subwords_batches(doc, self.config,
                                                     self.tokenizer)

        special_tokens = np.array([self.tokenizer.cls_token_id,
                                   self.tokenizer.sep_token_id,
                                   self.tokenizer.pad_token_id])
        subword_mask = ~(np.isin(subwords_batches, special_tokens))

        subwords_batches_tensor = torch.tensor(subwords_batches,
                                               device=self.config.device,
                                               dtype=torch.long)
        subword_mask_tensor = torch.tensor(subword_mask,
                                           device=self.config.device)

        # Obtain bert output for selected batches only
        attention_mask = (subwords_batches != self.tokenizer.pad_token_id)
        out = self.bert(
            subwords_batches_tensor,
            attention_mask=torch.tensor(
                attention_mask, device=self.config.device))

        # [n_subwords, bert_emb]
        return out[0][subword_mask_tensor]

    def _build_model(self):
        self.bert, self.tokenizer = bert.load_bert(self.config)
        #self.pw = PairwiseEncoder(self.config).to(self.config.device)
        self.pw_chunk = PairwiseEncoderChunk(self.config).to(self.config.device)
        self.chunk_aux = ChunkEncoder(self.config).to(self.config.device)

        bert_emb = self.bert.config.hidden_size
        #pair_emb = bert_emb * 3 + self.pw.shape
        #pair_emb_chunk = (bert_emb * 3+self.chunk_aux.shape)*3 + self.pw_chunk.shape
        pair_emb_chunk = (bert_emb)*3 + self.pw_chunk.shape

        # pylint: disable=line-too-long
        #self.a_scorer = AnaphoricityScorer(pair_emb, self.config).to(self.config.device)
        self.a_scorer_chunk = AnaphoricityScorerChunk(pair_emb_chunk, self.config).to(self.config.device)
        self.we = WordEncoder(bert_emb, self.config).to(self.config.device)
        #self.rough_scorer = RoughScorer(bert_emb, self.config).to(self.config.device)
        #self.rough_scorer_chunk = RoughScorerChunk(bert_emb*3+self.chunk_aux.shape, self.config).to(self.config.device)
        self.rough_scorer_chunk = RoughScorerChunk(bert_emb, self.config).to(self.config.device)
        #self.sp = SpanPredictor(bert_emb, self.config.sp_embedding_size).to(self.config.device)
        #self.sp_chunk = SpanPredictorChunk(bert_emb*3+self.chunk_aux.shape, self.config.sp_embedding_size).to(self.config.device)
        self.sp_chunk = SpanPredictorChunk(bert_emb, self.config.sp_embedding_size).to(self.config.device)

        """self.trainable: Dict[str, torch.nn.Module] = {
            "bert": self.bert, "we": self.we,
            "rough_scorer": self.rough_scorer,
            "pw": self.pw, "a_scorer": self.a_scorer,
            "sp": self.sp
        }"""
        self.trainable: Dict[str, torch.nn.Module] = {
            "bert": self.bert, "we": self.we,
            "rough_scorer_chunk": self.rough_scorer_chunk,
            "pw_chunk": self.pw_chunk, "a_scorer_chunk": self.a_scorer_chunk,
            #"sp": self.sp,
            "sp_chunk": self.sp_chunk, 
            #"chunk_aux": self.chunk_aux
        }

    def _build_optimizers(self):
        n_docs = len(self._get_docs(self.config.train_data))
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler.LambdaLR] = {}

        for param in self.bert.parameters():
            param.requires_grad = self.config.bert_finetune

        if self.config.bert_finetune:
            self.optimizers["bert_optimizer"] = torch.optim.Adam(
                self.bert.parameters(), lr=self.config.bert_learning_rate
            )
            self.schedulers["bert_scheduler"] = \
                transformers.get_linear_schedule_with_warmup(
                    self.optimizers["bert_optimizer"],
                    n_docs, n_docs * self.config.train_epochs
                )

        # Must ensure the same ordering of parameters between launches
        modules = sorted((key, value) for key, value in self.trainable.items()
                         if key != "bert")
        params = []
        for _, module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)

        self.optimizers["general_optimizer"] = torch.optim.Adam(
            params, lr=self.config.learning_rate)
        self.schedulers["general_scheduler"] = \
            transformers.get_linear_schedule_with_warmup(
                self.optimizers["general_optimizer"],
                0, n_docs * self.config.train_epochs
            )

    def _clusterize(self, doc: Doc, scores: torch.Tensor, top_indices: torch.Tensor):
        antecedents = scores.argmax(dim=1) - 1
        not_dummy = antecedents >= 0
        coref_span_heads = torch.arange(0, len(scores), device="cuda")[not_dummy]
        antecedents = top_indices[coref_span_heads, antecedents[not_dummy]]

        nodes = [GraphNode(i) for i in range(len(doc["cased_words"]))]
        for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
            nodes[i].link(nodes[j])
            assert nodes[i] is not nodes[j]

        clusters = []
        for node in nodes:
            if len(node.links) > 0 and not node.visited:
                cluster = []
                stack = [node]
                while stack:
                    current_node = stack.pop()
                    current_node.visited = True
                    cluster.append(current_node.id)
                    stack.extend(link for link in current_node.links if not link.visited)
                assert len(cluster) > 1
                clusters.append(sorted(cluster))
        return sorted(clusters)

    def _clusterize_chunk(self, doc: Doc, scores: torch.Tensor, top_indices: torch.Tensor):
        antecedents = scores.argmax(dim=1) - 1
        not_dummy = antecedents >= 0
        coref_span_heads = torch.arange(0, len(scores), device="cuda")[not_dummy]
        antecedents = top_indices[coref_span_heads, antecedents[not_dummy]]

        #nodes = [GraphNode(i) for i in range(len(doc["chunk_list"])+len(doc["extra_chunks"]))]
        nodes = [GraphNode(i) for i in range(len(doc["splitted_chunk"]))]
        for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
            nodes[i].link(nodes[j])
            assert nodes[i] is not nodes[j]

        clusters = []
        for node in nodes:
            if len(node.links) > 0 and not node.visited:
                cluster = []
                stack = [node]
                while stack:
                    current_node = stack.pop()
                    current_node.visited = True
                    cluster.append(current_node.id)
                    stack.extend(link for link in current_node.links if not link.visited)
                assert len(cluster) > 1
                clusters.append(sorted(cluster))
        return sorted(clusters)

    def _get_docs(self, path: str) -> List[Doc]:
        special_pronoun_list = ['his', 'their', 'its', 'my', 'your', 'her', 'our']
        if path not in self._docs:
            basename = os.path.basename(path)
            model_name = self.config.bert_model.replace("/", "_")

            cache_filename = f"{model_name}_{basename}.pickle"
            if "train" in basename:
                cache_name = "train_tokenized_w_chunks.pt" #f"train_chunks_{basename}.pt"
                preprocessed_name = f"train_preprocessed.pt"
            elif "development" in basename:
                cache_name = "dev_tokenized_w_chunks.pt"
                preprocessed_name = "dev_preprocessed.pt"
            elif "test" in basename:
                cache_name = "test_tokenized_w_chunks.pt"
                preprocessed_name = "test_preprocessed.pt"

            if os.path.exists(preprocessed_name):
                with open(preprocessed_name, mode="rb") as cache_f:
                    self._docs[path] = pickle.load(cache_f)
                    return self._docs[path]
            """if os.path.exists(cache_filename):
                with open(cache_filename, mode="rb") as cache_f:
                    self._docs[path] = pickle.load(cache_f)
            else:
                self._docs[path] = self._tokenize_docs(path)
                with open(cache_filename, mode="wb") as cache_f:
                    pickle.dump(self._docs[path], cache_f)"""
            if os.path.exists(cache_name):
                with open(cache_name, mode="rb") as cache_f:
                    self._docs[path] = pickle.load(cache_f)
            for doc_num, item in tqdm(enumerate(self._docs[path]), total=len(self._docs[path]), desc="procssing extra chunks"):
                self._docs[path][doc_num]['extra_chunks'] = []
                if "\'" in self._docs[path][doc_num]['cased_words'] or "\'s" in self._docs[path][doc_num]['cased_words']:
                    for i, e in enumerate(self._docs[path][doc_num]['cased_words']):
                        if (e == "\'s" or e == "\'") and (self._docs[path][doc_num]["conll_bound"][i-1] == 1 \
                                                          and self._docs[path][doc_num]["conll_bound"][i] == 0):
                            self._docs[path][doc_num]["conll_bound"][i-1] = 0
                            self._docs[path][doc_num]["conll_bound"][i] = 1
                temp_chunk = []
                conll_chunk_boundary = []
                for i, bound in enumerate(item['conll_bound']):
                    if i == 0:
                        temp_chunk.append(i)
                        if bound == 1:
                            conll_chunk_boundary.append(temp_chunk)
                            temp_chunk = []
                        continue
                    if bound == 1:
                        temp_chunk.append(i)
                        conll_chunk_boundary.append(temp_chunk)
                        temp_chunk = []
                    elif bound == 0:
                        temp_chunk.append(i)
                conll_chunk_boundary = [[e[0], e[-1]+1] for e in conll_chunk_boundary]
                self._docs[path][doc_num]['chunk_list'] = conll_chunk_boundary
                splitted_chunk = []
                for chunk in conll_chunk_boundary:
                    if chunk[1] - chunk[0] == 1:
                        splitted_chunk.append(chunk)
                        continue
                    chunk_text = item['cased_words'][chunk[0]:chunk[1]]
                    chunk_text = [c.lower() for c in chunk_text]
                    temp_split = []
                    is_splitted = False
                    if chunk_text == "\'s":
                        print()
                    for pronoun in special_pronoun_list:
                        if pronoun in chunk_text:
                            for i, e in enumerate(chunk_text):
                                if pronoun == e:
                                    self._docs[path][doc_num]["extra_chunks"].append([chunk[0]+i,chunk[0]+i+1])
                                    temp_split.append([chunk[0]+i,chunk[0]+i+1])
                                    is_splitted = True
                    if is_splitted:
                        if len(temp_split) == 1:
                            before_start = temp_split[0][0] - chunk[0]
                            after_end = chunk[1] - temp_split[0][1]
                            if before_start == 0:
                                splitted_chunk.append(temp_split[0])
                                splitted_chunk.append([temp_split[0][1], chunk[1]])
                            elif after_end == 0:
                                splitted_chunk.append([chunk[0], temp_split[0][0]])
                                splitted_chunk.append(temp_split[0])
                            else:
                                splitted_chunk.append([chunk[0], temp_split[0][0]])
                                splitted_chunk.append(temp_split[0])
                                splitted_chunk.append([temp_split[0][1], chunk[1]])
                        else:
                            prev_end = -1
                            for i, split in enumerate(temp_split):
                                if i == 0:
                                    if split[0] - chunk[0] >0:
                                        splitted_chunk.append([chunk[0], split[0]])
                                        prev_end = split[1]
                                    else:
                                        splitted_chunk.append(split)
                                        prev_end = split[1]
                                elif i == len(temp_split) -1:
                                    if split[0] - prev_end > 0:
                                        splitted_chunk.append([prev_end, split[0]])
                                        splitted_chunk.append(split)
                                    if chunk[1] - split[1] > 0:
                                        splitted_chunk.append([split[1], chunk[1]])
                                else:
                                    if split[0] - prev_end > 0:
                                        splitted_chunk.append([prev_end, split[0]])
                                        splitted_chunk.append(split)
                                        prev_end = split[1]
                                    else:
                                        splitted_chunk.append(split)
                                        prev_end = split[1]

                    else:
                        splitted_chunk.append(chunk)
                prev = 0
                for split in splitted_chunk:
                    assert split[0] == prev
                    prev = split[1]
                self._docs[path][doc_num]["splitted_chunk"] = splitted_chunk
                assert len(self._docs[path][doc_num]["chunk_list"]) == self._docs[path][doc_num]["conll_bound"].count(1)
                
                subword_mask = self.slice2label(item['word2subword'])

                chunk_subword = self.insert_subwords(subword_mask, self.slice2label(splitted_chunk))
                sentid_subword = self.insert_subwords(subword_mask, item['sent_id'])

                word_clusters_subword = [[self.index_after_subword(subword_mask, e) for e in clus] for clus in item['word_clusters']]
                span_clusters_subword = [[[self.index_after_subword(subword_mask, e) for e in span] for span in clus] for clus in item['span_clusters']]
                head2span_subword = [[self.index_after_subword(subword_mask, e) for e in clus] for clus in item['head2span']]
                
                item['chunk_subword'] = self.label2slice(chunk_subword)
                item['sentid_subword'] = sentid_subword
                item['word_clusters_subword'] = word_clusters_subword
                item['span_clusters_subword'] = span_clusters_subword
                item['head2span_subword'] = head2span_subword

                single_chunk_pos = []
                nonsingle_chunk_pos = []
                sent_id_chunk = []
                """for i, (s, e) in enumerate(item["chunk_list"]):
                    if e-s == 1:
                        single_chunk_pos.append([s,e])
                    else:
                        nonsingle_chunk_pos.append([s,e])
                    sent_id_chunk.append(item["sent_id"][s])
                for i, pos in enumerate(item["extra_chunks"]):
                    single_chunk_pos.append(pos)
                    sent_id_chunk.append(item["sent_id"][pos[0]])"""
                for i, (s, e) in enumerate(item["splitted_chunk"]):
                    sent_id_chunk.append(item["sent_id"][s])
                self._docs[path][doc_num]["sent_id_chunk"] = sent_id_chunk
                """cluster_ids_chunk = []
                for i, cluster in enumerate(self._docs[path][doc_num]['span_clusters']):
                    cluster_ids_chunk.append([0]*(len(single_chunk_pos)+len(nonsingle_chunk_pos)))
                    for span in cluster:
                        for ic, chunk in enumerate(single_chunk_pos):
                            if chunk[0] == span[0] and chunk[1] == span[1]:
                                cluster_ids_chunk[-1][ic] = i+1
                        for ic, chunk in enumerate(nonsingle_chunk_pos):
                            if chunk[0] == span[0] and chunk[1] == span[1]:
                                cluster_ids_chunk[-1][ic+len(single_chunk_pos)] = i+1"""
                cluster_ids_chunk = []
                """for i, cluster in enumerate(self._docs[path][doc_num]['span_clusters']):
                    cluster_ids_chunk.append([0]*(len(self._docs[path][doc_num]['chunk_list'])+len(self._docs[path][doc_num]['extra_chunks'])))
                    for span in cluster:
                        for ic, chunk in enumerate(self._docs[path][doc_num]['chunk_list']):
                            if chunk[0] >= span[0] and chunk[1] <= span[1]:
                                cluster_ids_chunk[-1][ic] = i+1
                        for ic, chunk in enumerate(self._docs[path][doc_num]['extra_chunks']):
                            if chunk[0] >= span[0] and chunk[1] <= span[1]:
                                cluster_ids_chunk[-1][ic+len(self._docs[path][doc_num]['chunk_list'])] = i+1"""
                for i, cluster in enumerate(self._docs[path][doc_num]['span_clusters']):
                    cluster_ids_chunk.append([0]*(len(self._docs[path][doc_num]['splitted_chunk'])))
                    for span in cluster:
                        for ic, chunk in enumerate(self._docs[path][doc_num]['splitted_chunk']):
                            if chunk[0] >= span[0] and chunk[1] <= span[1]:
                                cluster_ids_chunk[-1][ic] = i+1
                self._docs[path][doc_num]['cluster_ids_chunk'] = cluster_ids_chunk
                self._docs[path][doc_num]['single_chunk_pos'] = single_chunk_pos
                self._docs[path][doc_num]['nonsingle_chunk_pos'] = nonsingle_chunk_pos

                chunk_head = []
                #temp_chunk_list = item['chunk_list'].copy()
                #temp_chunk_list.extend(item['extra_chunks'])
                """temp_chunk_list = item['splitted_chunk'].copy()
                for e in item['head2span']:
                    not_found = True
                    for i, chunk in enumerate(temp_chunk_list):
                        if chunk[0] >= e[1] and chunk[1] <= e[2]:
                            chunk_head.append([i, e[1], e[2]])
                            not_found = False
                    if not_found:
                        for i, chunk in enumerate(temp_chunk_list):
                            if e[0] <= chunk[1] and e[0] >= chunk[0]:
                                chunk_head.append([i, e[1], e[2]])
                chunk_head.sort()
                chunk_head = list(e for e,_ in itertools.groupby(chunk_head))
                if len(chunk_head) >0:
                    heads, starts, ends = zip(*chunk_head)
                    spans = [[s, e] for s, e in zip(starts, ends)]
                    count_heads = Counter(heads)
                    dup_heads = []
                    for head, c in count_heads.items():
                        if c>1:
                            for head_tup in chunk_head:
                                if head_tup[0] == head:
                                    dup_heads.append(head_tup)
                    for duphead in dup_heads:
                        if spans.count(duphead[1:]) > 1:
                            chunk_head.remove(duphead)
                chunk_tok_head = []
                for ch in chunk_head:
                    temp = temp_chunk_list[ch[0]]
                    for c in list(range(temp[0], temp[1])):
                        chunk_tok_head.append([c, ch[1], ch[2]])
                self._docs[path][doc_num]['chunk_head'] = chunk_tok_head"""
                        

                temp_chunk_list = item['splitted_chunk'].copy()
                for e in item['head2span']:
                    not_found = True
                    head_candidate = []
                    for i, chunk in enumerate(temp_chunk_list):
                        if chunk[0] >= e[1] and chunk[1] <= e[2]:
                            temp_start_pos = -1
                            temp_end_pos = -1
                            for ic, c in enumerate(temp_chunk_list):
                                if e[1] == c[0]:
                                    temp_start_pos = ic
                            for ic, c in enumerate(temp_chunk_list):
                                if e[2] == c[1]:
                                    temp_end_pos = ic
                            if temp_start_pos != -1 and temp_end_pos != -1:
                                head_candidate.append([i, temp_start_pos, temp_end_pos])
                                not_found = False
                    if len(head_candidate) > 1:
                        for head in head_candidate:
                            chunk = temp_chunk_list[head[0]]
                            if chunk[0] <= e[0] and chunk[1] >= e[0]:
                                chunk_head.append(head)
                    elif len(head_candidate) == 1:
                        chunk_head.append(head_candidate[0])
                    #if not_found:
                    #    for i, chunk in enumerate(temp_chunk_list):
                    #        if e[0] <= chunk[1] and e[0] >= chunk[0]:
                    #            for c in list(range(chunk[0], chunk[1]))[:-1]:
                    #                chunk_head.append([c, e[1], e[2]])
                chunk_head.sort()
                chunk_head = list(e for e,_ in itertools.groupby(chunk_head))
                self._docs[path][doc_num]['chunk_head'] = chunk_head
                   
        return self._docs[path]
    
    @staticmethod
    def slice2label(slices: list[list]):
        starts = [0] * slices[-1][1]
        for s in slices:
            starts[s[0]] = 1
        return starts
    
    @staticmethod
    def label2slice(labels: list):
        slices = []
        start_index = None
        for i, s in enumerate(labels):
            if s == 1 and start_index is None:
                start_index = i
            elif s == 0 and start_index is not None:
                slices.append([start_index, i])
                start_index = None
        if start_index is not None:
            slices.append([start_index, len(labels)])
        return slices


    @staticmethod
    def insert_subwords(subword_mask, input_seq):
        output = []
        input_seq_iter = iter(input_seq)
        prev_ = next(input_seq_iter)
        for i, sw in enumerate(subword_mask):
            if sw == 1:
                try:
                    prev_ = next(input_seq_iter)
                except StopIteration:
                    prev_ = input_seq[-1]
            output.append(prev_)
        return output


    @staticmethod
    def index_after_subword(starts, n):
        if n == 0:
            return n
        ones = (i for i, s in enumerate(starts) if s == 1)
        try:
            return next(itertools.islice(ones, n-1, n))
        except StopIteration:
            return -1



    @staticmethod
    def _load_config(config_path: str,
                     section: str) -> Config:
        config = toml.load(config_path)
        default_section = config["DEFAULT"]
        current_section = config[section]
        unknown_keys = (set(current_section.keys())
                        - set(default_section.keys()))
        if unknown_keys:
            raise ValueError(f"Unexpected config keys: {unknown_keys}")
        return Config(section, **{**default_section, **current_section})

    def _set_training(self, value: bool):
        self._training = value
        for module in self.trainable.values():
            module.train(self._training)

    def _tokenize_docs(self, path: str) -> List[Doc]:
        self.logger.info(f"Tokenizing documents at {path}...", flush=True)
        out: List[Doc] = []
        filter_func = TOKENIZER_FILTERS.get(self.config.bert_model,
                                            lambda _: True)
        token_map = TOKENIZER_MAPS.get(self.config.bert_model, {})
        with jsonlines.open(path, mode="r") as data_f:
            for doc in data_f:
                doc["span_clusters"] = [[tuple(mention) for mention in cluster]
                                   for cluster in doc["span_clusters"]]
                word2subword = []
                subwords = []
                word_id = []
                for i, word in enumerate(doc["cased_words"]):
                    tokenized_word = (token_map[word]
                                      if word in token_map
                                      else self.tokenizer.tokenize(word))
                    tokenized_word = list(filter(filter_func, tokenized_word))
                    word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
                    subwords.extend(tokenized_word)
                    word_id.extend([i] * len(tokenized_word))
                doc["word2subword"] = word2subword
                doc["subwords"] = subwords
                doc["word_id"] = word_id
                out.append(doc)
        self.logger.info("Tokenization OK", flush=True)
        return out
