import os
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
import torch.nn.functional as F
import torch.distributed as dist
import tqdm
import wandb
from accelerate import PartialState, Accelerator
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"



@dataclass
class FinetuneConfig:
    vla_path: str = "openvla-7b"  # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/calvin_abc")               # Path to CALVIN dataset directory
    dataset_name: str = "calvin"                                    # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                             # Fine-tuning batch size
    max_steps: int = 1000000                                         # Max number of fine-tuning steps
    epochs: int = 50                                                 # Num of training epoches
    save_steps: int = 20000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                      # Fine-tuning learning rate
    grad_accumulation_steps: int = 4                                 # Gradient accumulation steps
    image_aug: bool = True                                           # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                               # Dataloader shuffle buffer size (can reduce if OOM)
    save_epochs = 5                                                  # Ckpts save schedual

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance
    # Tracking Parameters
    wandb_project: str = "robodual"                                 # Name of W&B project to log to (use default!)
    wandb_entity: str = "opendrivelab"                              # Name of entity to log under




@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"openvla-{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"

    exp_id += '_OpenVLA-Generalist'

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()


    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]

    trainable_total_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)

    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)


    from prismatic.vla.datasets import DiskCalvinDataset
    vla_dataset = DiskCalvinDataset(
        datasets_dir=cfg.data_root_dir / "training",
        window_size=8,              # Number of frames to load
        action_chunking_size=0,     # Action chunking size of specialist model, set to '0' when training the generalist
        partial_data=1,             # Value range: [0, 1], indicating the proportion of data to load
        sampling_step=1,            # Frame interval
        action_tokenizer = action_tokenizer,
        base_tokenizer = processor.tokenizer,
        image_transform = processor.image_processor.apply_transform,
        prompt_builder_fn = PurePromptBuilder,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if accelerator.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=None,
        collate_fn=collator,
        pin_memory=True,
        num_workers=16, 
    )

    vla, optimizer, dataloader = accelerator.prepare(
        vla, optimizer, dataloader
    )

    # Initialize Logging =>> W&B
    if accelerator.is_main_process:
        # wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")
        wandb.init(name=f"ft+{exp_id}")

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        global_step = 0
        for e in range(cfg.epochs):
            progress.set_description("Epoch " + str(e+1))
            for step_idx, batch in enumerate(dataloader):
                global_step += 1
                with accelerator.autocast():
                    input_ids = batch["input_ids"].to(device_id)

                    output: CausalLMOutputWithPast = vla(
                        input_ids=input_ids,
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                        output_hidden_states = True,        # Return intermediate tokens of all layers
                    )
                    act_loss = output.loss


                loss = act_loss / cfg.grad_accumulation_steps
                accelerator.backward(loss)

                action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                progress.set_postfix(act_loss='{:.4f}'.format(act_loss.item()))

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Push Metrics to W&B (every 10 steps)
                if accelerator.is_main_process and step_idx % 10 == 0:
                    wandb.log(
                        {"train_loss": loss, "action_accuracy": action_accuracy, "l1_loss": action_l1_loss}, step=global_step
                    )

                # Optimizer Step
                if (step_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()
                    

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if global_step > 0 and (e + 1) % cfg.save_epochs == 0:
                if accelerator.is_main_process:
                    print(f"Saving Model Checkpoint for Step {step_idx}")

                    # We save all ckpts (instead of just the last one)
                    tmp_exp_name = exp_id + f'_epoch-{e}'
                    run_dir, adapter_dir = cfg.run_root_dir / tmp_exp_name, cfg.adapter_tmp_dir / tmp_exp_name
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                    # Merge LoRA weights into model backbone for faster inference
                    #   =>> TODO (kpertsch, siddk) :: This is inefficient; probably want to do this post-hoc...
                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        merged_vla.save_pretrained(run_dir)

                # Block on Main Process Checkpointing
                dist.barrier()


if __name__ == "__main__":
    finetune()
