from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer, PreTrainedTokenizer
from trl import DataCollatorForCompletionOnlyLM


def formatting_prompts_func(input: dict[str, list[str]]) -> list[str]:
    """
    Format the prompt for the model with the instruction and response. Adapted from flower
    FlowerTune example:
    https://github.com/adap/flower/blob/main/examples/flowertune-llm/flowertune_llm/dataset.py.

    Args:
        input (dict[str, list[str]]): A dictionary containing the instruction and response.

    Returns:
        (list[str]): A list of formatted prompts.
    """
    output_texts = []

    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    message = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    )
    for i in range(len(input["instruction"])):
        text = f"{message}\n### Instruction:\n{input['instruction'][i]}\n### Response: {input['response'][i]}"
        output_texts.append(text)
    return output_texts


def get_alpaca_tokenizer_and_data_collator(
    model_name: str,
) -> tuple[PreTrainedTokenizer, DataCollatorForCompletionOnlyLM]:
    """
    Get tokenizer and data collator for the model. Adapted from flower FlowerTune example:
    https://github.com/adap/flower/blob/main/examples/flowertune-llm/flowertune_llm/dataset.py.

    Args:
        model_name (str): Model name that is supported by the Hugging Face Transformers library.

    Returns:
        (tuple[PreTrainedTokenizer, DataCollatorForCompletionOnlyLM]): The tokenizer and data collator.
    """
    # From: https://huggingface.co/docs/trl/en/sft_trainer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    return tokenizer, data_collator


def load_data(partition_id: int, num_partitions: int, dataset_name: str) -> Dataset:
    """
    Load partitioned data using Flower datasets. We utilize the IID partitioner to split the data and pass
    it into the FederatedDataset. This wrapper around Hugging Face Datasets simplifies data partitioning for
    federated learning. Adapted from flower FlowerTune example:
    https://github.com/adap/flower/blob/main/examples/flowertune-llm/flowertune_llm/dataset.py.

    Args:
        partition_id (int): The partition id.
        num_partitions (int): The number of partitions.
        dataset_name (str): The dataset name.

    Returns:
        (Dataset): The partitioned data

    """
    partitioner = IidPartitioner(num_partitions=num_partitions)
    federated_dataset = FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": partitioner},
    )
    client_trainset = federated_dataset.load_partition(partition_id, "train")
    return client_trainset.rename_column("output", "response")
