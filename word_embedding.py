from typing import Any

import torch
from torch import nn
from torch.distributions.uniform import Uniform
import lightning as L
from torch.optim import Adam


class WordEmbeddings(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        network_width: int = 2,
    ):
        super().__init__()
        min_thresh = -0.5
        max_thresh = 0.5
        # set weights with appropriate width and length
        self.vocab_size = vocab_size
        self.network_width = network_width
        for width in range(network_width):
            for vocab_item_weight in range(vocab_size):
                setattr(
                    self,
                    f"input_weight_{vocab_item_weight}_{width}",
                    nn.Parameter(Uniform(min_thresh, max_thresh).sample()),
                )
                setattr(
                    self,
                    f"output_weight_{vocab_item_weight}_{width}",
                    nn.Parameter(Uniform(min_thresh, max_thresh).sample()),
                )
        self.loss = nn.CrossEntropyLoss()

        pass

    def forward(self, input_tensor):
        inputs = []
        input_tensor = input_tensor[0]
        for width in range(self.network_width):
            input_item_weight = 0.0
            for vocab_item_weight in range(self.vocab_size):
                input_item_weight = input_item_weight + (
                    input_tensor[vocab_item_weight]
                    * getattr(self, f"input_weight_{vocab_item_weight}_{width}")
                )
            inputs.append(input_item_weight)
        outputs = []
        for width_id, width in enumerate(range(self.vocab_size)):
            output_item_weight = 0.0
            for vocab_item_weight_id, vocab_item_weight in enumerate(inputs):
                output_item_weight = output_item_weight + vocab_item_weight * getattr(
                    self, f"input_weight_{width_id}_{vocab_item_weight_id}"
                )
            outputs.append(output_item_weight)
        output_pre_softmax = torch.stack(outputs)

        return output_pre_softmax

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i[0])
        return loss
