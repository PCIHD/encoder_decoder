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


class WordnPositionalEmbeddings(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        network_width: int = 2,
    ):
        super().__init__()
        min_thresh = -0.5
        max_thresh = 0.5

        # add positional encoding methods as a list
        self.positional_encoders = get_positional_encoders(network_width)
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
                # positional encoding
                # for the nth token take frequency as n+1
                frequency = 1 * vocab_item_weight
                positional_encoding = self.positional_encoders[width](
                    torch.tensor((frequency) * width)
                )
                input_item_weight = input_item_weight + positional_encoding
            inputs.append(input_item_weight)
        outputs = []
        for width_id, width in enumerate(range(self.vocab_size)):
            output_item_weight = 0.0
            for vocab_item_weight_id, vocab_item_weight in enumerate(inputs):
                output_item_weight = output_item_weight + vocab_item_weight * getattr(
                    self, f"output_weight_{width_id}_{vocab_item_weight_id}"
                )
                # # positional encoding
                # # for the nth token take frequency as n+1
                # frequency = 1 * vocab_item_weight_id
                # positional_encoding = self.positional_encoders[vocab_item_weight_id](
                #     torch.tensor((frequency) * width_id)
                # )
                # output_item_weight = output_item_weight + positional_encoding

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


def get_positional_encoders(network_width):
    encoders = [torch.sin, torch.cos]
    # based on the network width provide alternating sin and cos functions as encoders. Frequency management is done in forward method
    return [
        encoders[network_with_element % 2]
        for network_with_element in range(network_width)
    ]


class WordnPositionalSelfAttentionEmbeddings(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        network_width: int = 2,
    ):
        super().__init__()
        min_thresh = -0.5
        max_thresh = 0.5

        # add positional encoding methods as a list
        self.positional_encoders = get_positional_encoders(network_width)
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

                setattr(
                    self,
                    f"key_weight_{vocab_item_weight}_{width}",
                    nn.Parameter(Uniform(min_thresh, max_thresh).sample()),
                )

                setattr(
                    self,
                    f"query_weight_{vocab_item_weight}_{width}",
                    nn.Parameter(Uniform(min_thresh, max_thresh).sample()),
                )

                setattr(
                    self,
                    f"value_weight_{vocab_item_weight}_{width}",
                    nn.Parameter(Uniform(min_thresh, max_thresh).sample()),
                )
        self.loss = nn.CrossEntropyLoss()

        pass

    def forward(self, input_tensor):
        inputs = []
        input_tensor = input_tensor[0]
        for width in range(self.network_width):
            input_item_weight_vector = []
            for vocab_item_weight in range(self.vocab_size):
                input_item_weight = torch.tensor(0.0)
                input_item_weight = input_item_weight + (
                    input_tensor[vocab_item_weight]
                    * getattr(self, f"input_weight_{vocab_item_weight}_{width}")
                )
                # positional encoding
                # for the nth token take frequency as n+1
                frequency = 1 * vocab_item_weight
                positional_encoding = self.positional_encoders[width](
                    torch.tensor((frequency) * width)
                )
                input_item_weight = input_item_weight + positional_encoding
                input_item_weight_vector.append(input_item_weight)
            inputs.append(input_item_weight_vector)
        query_vector = []
        key_vector = []
        value_vector = []
        for input_id, input_value in enumerate(inputs):
            key = []
            query = []
            value = []
            for vocab_size_value in range(self.vocab_size):
                key.append(
                    input_value[vocab_size_value]
                    * getattr(self, f"key_weight_{vocab_size_value}_{input_id}")
                )
                query.append(
                    input_value[vocab_size_value]
                    * getattr(self, f"query_weight_{vocab_size_value}_{input_id}")
                )
                value.append(
                    input_value[vocab_size_value]
                    * getattr(self, f"value_weight_{vocab_size_value}_{input_id}")
                )
            query_vector.append(query)
            key_vector.append(key)
            value_vector.append(value)
        query_vector = torch.tensor(query_vector).to("cuda:0")
        key_vector = torch.tensor(key_vector).to("cuda:0")
        value_vector = torch.tensor(value_vector).to("cuda:0")

        attention_values = self.calculate_attention(
            query_vector, key_vector, value_vector
        )
        inputs = torch.tensor(inputs).to("cuda:0")
        output_values = inputs + attention_values
        outputs = []
        for width_id, width in enumerate(range(self.vocab_size)):
            output_item_weight = torch.tensor(0.0)
            for vocab_item_weight_id, vocab_item_weight in enumerate(output_values):
                output_item_weight = output_item_weight + vocab_item_weight[
                    width_id
                ] * getattr(self, f"output_weight_{width_id}_{vocab_item_weight_id}")

            outputs.append(output_item_weight)

        output_pre_softmax = torch.stack(outputs)
        return output_pre_softmax

    def calculate_attention(self, query_vector, key_vector, value_vector):
        qk_t = torch.matmul(query_vector, torch.transpose(key_vector, 1, 0))
        qk_t_div_v = torch.div(qk_t, torch.sqrt(torch.tensor(self.vocab_size)))
        softmaxed = torch.softmax(qk_t_div_v, dim=0)
        attention = torch.matmul(softmaxed, value_vector)
        return attention

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i[0])
        return loss
