""" Describes PairwiseEncodes, that transforms pairwise features, such as
distance between the mentions, same/different speaker into feature embeddings
"""
from typing import List

import torch

from coref.config import Config
from coref.const import Doc

class ChunkEncoder(torch.nn.Module):
    """ A Pytorch module to obtain feature embeddings for pairwise features

    Usage:
        encoder = PairwiseEncoder(config)
        pairwise_features = encoder(pair_indices, doc)
    """
    def __init__(self, config: Config):
        super().__init__()
        emb_size = config.embedding_size

        self.genre2int = {g: gi for gi, g in enumerate(["bc", "bn", "mz", "nw",
                                                        "pt", "tc", "wb"])}
        self.genre_emb = torch.nn.Embedding(len(self.genre2int), emb_size)

        # each position corresponds to a bucket:
        #   [(0, 2), (2, 3), (3, 4), (4, 5), (5, 8),
        #    (8, 16), (16, 32), (32, 64), (64, float("inf"))]
        # [1, 2-3, 4-7, >=8]
        self.distance_emb = torch.nn.Embedding(4, emb_size)

        #self.pos_emb = torch.nn.Embedding(2000, emb_size)

        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.shape = emb_size * 2  # genre, distance, speaker\

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.genre_emb.parameters()).device

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                doc: Doc,
                chunks_length:List,
                start_pos:torch.Tensor) -> torch.Tensor:
        #word_ids = torch.arange(0, len(doc["cased_words"]), device=self.device)
        #speaker_map = torch.tensor(self._speaker_map(doc, chunks_pos), device=self.device)
        #pos_emb = self.pos_emb(torch.arange(start_pos.size(0), device=self.device))
        
        # bucketing the distance (see __init__())
        log_distance = chunks_length.to(torch.float).log2().floor_()
        log_distance = log_distance.clamp_max_(max=3).to(torch.long)
        distance = self.distance_emb(log_distance)

        genre = torch.tensor(self.genre2int[doc["document_id"][:2]]).to(self.device).expand_as(start_pos)
        genre = self.genre_emb(genre)

        return self.dropout(torch.cat((distance, genre), dim=1))

