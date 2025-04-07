"""
specialist_train.py

Simple script for efficient training of the specialsit policy.

"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torchtune
import draccus
import torch
import torch.distributed as dist
import tqdm
import wandb
from ema_pytorch import EMA
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
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from einops import rearrange, repeat

from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from diffusers.schedulers import DPMSolverMultistepScheduler

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DualSystem(torch.nn.Module):
    def __init__(self, slow_system, fast_system, action_tokenizer, adapter = None, freeze_slow = True):
        super().__init__()
        self.slow_system = slow_system
        self.fast_system = fast_system
        self.adapter = adapter
        self.action_tokenizer = action_tokenizer

        # Set power = 3/4 to get faster convergence
        self.ema_fast_system = EMA(self.fast_system, power = 0.75, beta = 0.9999, update_every = 1).to(fast_system.device)

        if freeze_slow:
            self.slow_system.requires_grad_(False)


    def forward(self, batch):
        with torch.no_grad():
            slow_output = self.slow_forward(batch)
        loss = self.fast_forward(batch, slow_output)
        return slow_output, loss


    def slow_forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = self.slow_system(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states = True,        # Return intermediate tokens of all layers
            )
        return output
    

    def fast_forward(self, batch, slow_output):
        # Get diffusion condition
        action_logits = slow_output.logits[:, self.slow_system.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        mask = batch["labels"][:, 1:].to(action_preds.device) > self.action_tokenizer.action_token_begin_idx

        # Discrete action outputs
        continuous_actions_pred = []
        for b in range(mask.shape[0]):
            continuous_actions_pred.append(torch.tensor(self.action_tokenizer.decode_token_ids_to_actions(action_preds[b, mask[b]].cpu().numpy())))
        continuous_actions_pred = torch.stack(continuous_actions_pred).to(batch['raw_action'].device)
        continuous_actions_pred = rearrange(continuous_actions_pred, 'b (f n) -> b f n', n=7)
        
        ref_actions = []
        for idx, ref_action in enumerate(continuous_actions_pred):
            zero_actions = torch.zeros((8, 7))
            num_cond_actions = batch['valid_cond_mask'][idx].sum().int()
            zero_actions[:num_cond_actions] = ref_action[-num_cond_actions:]
            ref_actions.append(zero_actions)
        ref_actions = torch.stack(ref_actions).to(action_logits.device)

        # Task and action latents
        latent_action_tokens = slow_output.hidden_states[-1][:, self.slow_system.vision_backbone.featurizer.patch_embed.num_patches : ]


        # Run specialist policy
        obs = (batch["pixel_values_dp"].to(torch.float), batch["prev_pixel_values_dp"].to(torch.float))
        dp_loss = self.fast_system.compute_loss(trajectory = batch['raw_action'], 
                                                ref_action = ref_actions.to(torch.float),
                                                hist_action = batch['hist_action'].to(torch.float),
                                                action_cond = latent_action_tokens.to(torch.float),
                                                obs = obs,
                                                depth_obs = batch["depth_image"].to(torch.float),
                                                gripper_obs = (batch["gripper_image"].to(torch.float), batch["depth_gripper"].to(torch.float)),
                                                tactile_obs = batch["tactile_image"].to(torch.float),
                                                lang= batch["lang"],
                                                proprio = batch["proprio"],
                                                decoupled_loss=False)

        return dp_loss


    def sample_action(self, batch):
        slow_output = self.slow_forward(batch)

        ### Get diffusion condition
        action_logits = slow_output.logits[:, self.slow_system.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        mask = batch["labels"][:, 1:].to(action_preds.device) > self.action_tokenizer.action_token_begin_idx

        # Discrete action outputs
        continuous_actions_pred = []
        for b in range(mask.shape[0]):
            continuous_actions_pred.append(torch.tensor(self.action_tokenizer.decode_token_ids_to_actions(action_preds[b, mask[b]].cpu().numpy())))
        continuous_actions_pred = torch.stack(continuous_actions_pred).to(batch['raw_action'].device)
        continuous_actions_pred = rearrange(continuous_actions_pred, 'b (f n) -> b f n', n=7)
        
        ref_actions = []
        for idx, ref_action in enumerate(continuous_actions_pred):
            zero_actions = torch.zeros((8, 7))
            num_cond_actions = batch['valid_cond_mask'][idx].sum().int()
            if num_cond_actions > 0:
                zero_actions[:num_cond_actions] = ref_action[-num_cond_actions:]
            ref_actions.append(zero_actions)
        ref_actions = torch.stack(ref_actions).to(action_logits.device)

        # Task and action latents
        latent_action_tokens = slow_output.hidden_states[-1][:, self.slow_system.vision_backbone.featurizer.patch_embed.num_patches : ]


        obs = (batch["pixel_values_dp"].to(torch.float), batch["prev_pixel_values_dp"].to(torch.float))

        # Sample actions
        pred_action = self.ema_fast_system.ema_model.predict_action(ref_action = ref_actions.to(torch.float),
                                                                    hist_action = batch['hist_action'].to(torch.float),
                                                                    action_cond = latent_action_tokens.to(torch.float),
                                                                    obs = obs,
                                                                    depth_obs = batch["depth_image"].to(torch.float),
                                                                    gripper_obs = (batch["gripper_image"].to(torch.float), batch["depth_gripper"].to(torch.float)),
                                                                    tactile_obs = batch["tactile_image"].to(torch.float),
                                                                    lang= batch["lang"],
                                                                    proprio = batch["proprio"])

        return pred_action


    

        

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla-7b"                                    # Path to OpenVLA model 

    # Directory Paths
    data_root_dir: Path = Path("datasets/calvin_abc")               # Path to CALVIN dataset directory
    dataset_name: str = "calvin"                                    # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Specialist Model Configs
    action_chunk_size: int = 8                                      # Action chunking size (TODO:current dataloader only works for size=8)
    num_inference_steps: int = 5                                    # Sampling steps 
    cond_drop_chance: float = 0.1                                   # Classifier-free guidance
    with_depth: bool = True                                         # Use depth input
    with_gripper: bool = True                                       # Use gripper-view inputs (both RGB and depth)
    with_tactile: bool = False                                      # Use tactile input
    vision_encoder = 'DINO'                                         # Supports: [DINO, Theia]

    # Fine-tuning Parameters
    batch_size: int = 8                                             # Fine-tuning batch size
    max_steps: int = 100000                                         # Max number of fine-tuning steps
    epochs: int = 50                                                # Num of training epoches
    save_steps: int = 10000                                         # Interval for checkpoint saving
    learning_rate: float = 1e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = False                                          # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    # => CAUTION: Reduces memory but hurts performance
    freeze_slow = True                                              # We train the specialsit only for efficiency

    # Tracking Parameters
    wandb_project: str = "robodual"                                 # Name of W&B project to log to (use default!)
    wandb_entity: str = "opendrivelab"                              # Name of entity to log under




@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    # Configure Unique Experiment ID & Log Directory
    exp_id = (#{cfg.vla_path.split('/')[-1]}+
        f"{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"

    exp_id += "_DiffusionPolicy-Specialist" 

    if cfg.freeze_slow:
        exp_id += "+freeze-slow"
    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        # assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=False,
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

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)


    # Create specialist model
    scheduler = DDIMScheduler( num_train_timesteps = 100, beta_schedule = 'squaredcos_cap_v2', prediction_type="epsilon" )
    shape_meta = {'action' : {'shape': [7]}}
    policy = DiffusionDiTImagePolicy(   shape_meta = shape_meta,
                                        noise_scheduler = scheduler,
                                        n_action_steps=cfg.action_chunk_size, 
                                        num_inference_steps=cfg.num_inference_steps,
                                        vision_encoder=cfg.vision_encoder,
                                        cond_drop_chance=cfg.cond_drop_chance,
                                        progressive_noise=False,
                                        with_depth=args.with_depth,
                                        with_gripper=args.with_gripper,
                                        with_tactile=args.with_tactile,
                                        ).to(device_id)

    # Dual-system wrapping
    dual_system = DualSystem(slow_system = vla, fast_system = policy, action_tokenizer = action_tokenizer, freeze_slow = cfg.freeze_slow)

    trainable_total_params = sum(p.numel() for p in dual_system.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)

    dual_system = DDP(dual_system, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=False)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in dual_system.parameters() if param.requires_grad]

    groups = [{'params': dual_system.module.fast_system.parameters(), 'lr': 1e-4}]
    optimizer = AdamW(groups, lr=cfg.learning_rate, weight_decay=1e-3)

    scheduler = torchtune.modules.get_cosine_schedule_with_warmup(
                                                        optimizer = optimizer, 
                                                        num_warmup_steps = 1000, 
                                                        num_training_steps = cfg.max_steps,
                                                        num_cycles = 0.4,
                                                        )


    # Load CALVIN dataset
    from prismatic.vla.datasets import DiskCalvinDataset
    vla_dataset = DiskCalvinDataset(
        datasets_dir=cfg.data_root_dir / "training",
        window_size=8,              # Number of frames to load
        action_chunking_size=8,     # Action chunking size of specialist model, set to '0' when training the generalist
        partial_data=1,             # Value range: [0, 1], indicating the proportion of data to load
        sampling_step=1,            # Frame interval
        action_tokenizer = action_tokenizer,
        base_tokenizer = processor.tokenizer,
        image_transform = processor.image_processor.apply_transform,
        prompt_builder_fn = PurePromptBuilder,
        imagenet_norm=True if cfg.vision_encoder == 'DINO' else False,
    )


    # Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
        num_workers=16,  
    )

    dual_system, optimizer, dataloader, scheduler = accelerator.prepare(
        dual_system, optimizer, dataloader, scheduler
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        # wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")
        wandb.init(name=f"ft+{exp_id}", reinit=True)

    recent_train_losses = deque(maxlen=100)
    recent_action_l1 = deque(maxlen=100)


    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=True) as progress:
        dual_system.train()
        optimizer.zero_grad()
        global_step = 0
        for e in range(cfg.epochs):
            progress.set_description("Epoch " + str(e+1))
            if global_step > cfg.max_steps:
                print('Training complete...')
                break

            for step_idx, batch in enumerate(dataloader):
                global_step += 1
                batch["input_ids"] = batch["input_ids"].to(device_id)
                batch["attention_mask"] = batch["attention_mask"].to(device_id)
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
                batch["pixel_values_dp"] = batch["pixel_values_dp"].to(device_id)
                batch["prev_pixel_values_dp"] = batch["prev_pixel_values_dp"].to(device_id)
                batch["gripper_image"] = batch["gripper_image"].to(device_id)
                batch["depth_gripper"] = batch["depth_gripper"].to(device_id)
                batch["tactile_image"] = batch["tactile_image"].to(device_id)
                batch["depth_image"] = batch["depth_image"].to(device_id)
                batch['raw_action'] = batch['raw_action'].to(device_id)
                batch['hist_action']= batch['hist_action'].to(device_id)
                batch['proprio'] = batch['proprio'].to(device_id)


                slow_output, dp_loss = dual_system(batch)
                
                if cfg.freeze_slow:
                    loss = dp_loss
                else:
                    loss = dp_loss + slow_output.loss   # Co-train

                torch.nn.utils.clip_grad_norm_(dual_system.parameters(), max_norm=1.)

                recent_train_losses.append(dp_loss.item())
                
                # Backward!
                # loss.backward()
                accelerator.backward(loss)

                # Update ema model
                dual_system.module.ema_fast_system.update()

                progress.set_postfix(dp_loss='{:.4f}'.format(dp_loss.item()))


                # Push Metrics to W&B (every 10 steps)
                if distributed_state.is_main_process and step_idx % 10 == 0:

                    pred_action = dual_system.module.sample_action(batch)

                    # Compute validation loss
                    action_l1_loss_dp = torch.nn.functional.l1_loss(pred_action, batch['raw_action'])
                    recent_action_l1.append(action_l1_loss_dp.item())
                    action_l1_loss_dp_cur = torch.nn.functional.l1_loss(pred_action[:,0], batch['raw_action'][:,0])

                    smoothed_loss = sum(recent_train_losses) / len(recent_train_losses)
                    smoothed_l1_loss_dp = sum(recent_action_l1) / len(recent_action_l1)

                    wandb.log(
                        {"train_loss": loss, "smoothed_loss": smoothed_loss, 
                        "l1_loss_dp": action_l1_loss_dp, "l1_loss_dp_cur": action_l1_loss_dp_cur, "smoothed_l1_loss_dp": smoothed_l1_loss_dp,
                        "learning rate": optimizer.param_groups[0]['lr']}, step=global_step
                    )

                # Optimizer Step
                if (step_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()
                    scheduler.step()

                # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                if global_step > 0 and global_step % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {step_idx}")

                        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                        save_dir = adapter_dir if cfg.use_lora else run_dir

                        # Save Processor & Weights
                        if not cfg.freeze_slow:
                            processor.save_pretrained(run_dir)
                            dual_system.module.slow_system.save_pretrained(save_dir)

                        torch.save(dual_system.module.ema_fast_system.state_dict(), str(run_dir) + f'/fast_system_state_dict_ema-{global_step}.pt')

                        # Merge LoRA weights into model backbone for faster inference
                        if cfg.use_lora and not cfg.freeze_slow:
                            base_vla = AutoModelForVision2Seq.from_pretrained(
                                cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                            )
                            merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                            merged_vla = merged_vla.merge_and_unload()
                            merged_vla.save_pretrained(run_dir)

                    # Block on Main Process Checkpointing
                    dist.barrier()

                if global_step > cfg.max_steps:
                    break
                


if __name__ == "__main__":
    finetune()
