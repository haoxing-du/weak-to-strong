from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):
    """
    This class initializes the linear head to zeros
    """

    def __init__(self, name, linear_probe=False, **kwargs):
        config = AutoConfig.from_pretrained(name, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels
        lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.lm = lm
        self.transformer = lm.transformer
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))
        self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
            lm.lm_head.weight.dtype
        )
        torch.nn.init.normal_(self.score.weight, std=0.0)
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(name, **kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def forward(self, input_ids: torch.LongTensor):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        input_lens = (input_ids != 0).sum(dim=-1)
        transformer_outputs = self.transformer(input_ids)
        hidden_states = torch.stack(
            [transformer_outputs[0][i, input_lens[i] - 1, :] for i in range(len(input_lens))]
        )
        self.score.to(hidden_states.device)
        if self.linear_probe:
            hidden_states = hidden_states.detach()
        logits = self.score(hidden_states)
        return logits

def save_custom_model(model, save_directory):
    # Save the transformer model
    model.transformer.save_pretrained(f"{save_directory}/transformer")
    # Save the classification head separately
    torch.save(model.score.state_dict(), f"{save_directory}/classification_head.pt")

def load_custom_model(load_directory, name, linear_probe=False):
    # Load the transformer model
    transformer = AutoModelForCausalLM.from_pretrained(f"{load_directory}/transformer")
    # Initialize the custom model
    model = TransformerWithHead(name, linear_probe=linear_probe)
    model.transformer = transformer.transformer
    model.lm = transformer
    # Load the classification head
    model.score.load_state_dict(torch.load(f"{load_directory}/classification_head.pt"))
    return model