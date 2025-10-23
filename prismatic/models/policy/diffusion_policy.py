from typing import Dict
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from prismatic.models.policy.diffusion_transformer import DiT_Tiny_STA, DiT_Small_STA, DiT_Base_STA, DiT_Large_STA
from prismatic.models.policy.transformer_utils import MAPBlock



class DiffusionDiTImagePolicy(nn.Module):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,     
            n_action_steps = 4,         # Action chunking size for prediction 
            n_obs_steps = 1,            # Given only current obs.
            num_inference_steps=10,
            vision_encoder='DINO',
            with_depth=False,
            with_gripper=False,
            with_tactile=False,
            cond_drop_chance=0.,
            progressive_noise=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert vision_encoder in ['DINO', 'Theia']
        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        self.encoder_type = vision_encoder
        self.cond_drop_chance = cond_drop_chance
        self.with_cfg = True if cond_drop_chance > 0 else False
        self.guidance_scale = 3
        self.progressive_noise = progressive_noise

        # create diffusion model
        model = DiT_Tiny_STA(
                    num_actions = n_action_steps, 
                    vis_dim=384, 
                    num_obs_latents=8,
                    in_context=False, 
                    with_proprio=True, 
                    with_depth=with_depth,
                    with_gripper=with_gripper, 
                    with_tactile=with_tactile, 
                    with_hist_action_num=0,
        )

        encoder_embed_size = model.hidden_size if model.hidden_size == 256 else 384
        encoder_num_heads = 8 if encoder_embed_size == 256 else 6

        # Create Vision Encoder
        if vision_encoder == 'DINO':
            self.vision_encoder = timm.create_model(
                'vit_small_patch16_224.dino',
                pretrained=True,
                num_classes=0,  # remove classifier nn.Linear
                # pretrained_cfg={'file': 'vit_small_patch16_224.dino/pytorch_model.bin'}
            )
        elif vision_encoder == 'Theia':
            self.vision_encoder = AutoModel.from_pretrained("theaiinstitute/theia-small-patch16-224-cdiv")

        self.vision_encoder.requires_grad_(False)

        self.with_depth = with_depth
        if with_depth:
            from prismatic.models.policy.vit import ViT
            import torchvision.transforms as transforms
            depth_size = 224
            self.depth_resize = transforms.Resize(size = (depth_size, depth_size))
            self.depth_encoder = ViT(image_size = depth_size, patch_size = 16, dim = encoder_embed_size, depth = 6, heads = encoder_num_heads, mlp_dim = encoder_embed_size * 4, channels = 1)

        self.with_tactile = with_tactile
        if with_tactile:
            tactile_size = 128
            self.tactile_resize = transforms.Resize(size = (tactile_size, tactile_size))
            self.tactile_encoder = ViT(image_size = tactile_size, patch_size = 16, dim = encoder_embed_size, depth = 6, heads = encoder_num_heads, mlp_dim = encoder_embed_size * 4, channels = 6)

        self.with_gripper = with_gripper
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, condition_data, hist_action=None,
            local_cond=None, global_cond=None, lang=None, proprio=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        if self.progressive_noise:
            trajectory = self.gen_progressive_noise(condition_data).to(dtype=condition_data.dtype, device=condition_data.device)
        else:
            trajectory = torch.randn(
                size=condition_data.shape, 
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:

            # predict model output
            model_output = model(trajectory, t, cond=local_cond, context=global_cond[0],
                                 visual_embedding=global_cond[1], 
                                 depth_embedding=global_cond[2] if self.with_depth else None,
                                 gripper_embedding = (global_cond[3], global_cond[4]) if self.with_gripper else None,
                                 lang=lang, 
                                 hist_action=hist_action,
                                 proprio=proprio)
            
            if self.with_cfg:
                cond_mask = torch.zeros(trajectory.shape[0], 1,  device = trajectory.device).float()
                model_output_uncond = model(trajectory, t, cond=local_cond, context=global_cond[0],
                                 visual_embedding=global_cond[1], 
                                 depth_embedding=global_cond[2] if self.with_depth else None,
                                 gripper_embedding = (global_cond[3], global_cond[4]) if self.with_gripper else None,
                                 tactile_embedding = global_cond[5] if self.with_tactile else None,
                                 lang=lang, 
                                 hist_action=hist_action,
                                 proprio=proprio,
                                 cond_mask = cond_mask)
                model_output = model_output_uncond + self.guidance_scale * (model_output - model_output_uncond)

            # compute previous action: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample       

        return trajectory


    def predict_action(self, ref_action, action_cond, obs, depth_obs=None, gripper_obs=None, tactile_obs=None, lang=None, proprio=None, hist_action=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        B = proprio.shape[0]
        T = self.n_action_steps
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        if self.encoder_type == 'Theia':
            if isinstance(obs, tuple):
                visual_embedding = torch.stack([self.vision_encoder.forward_feature(image.permute(0,2,3,1) * 0.5 + 0.5) for image in obs], dim=1)
            else:
                visual_embedding = self.vision_encoder.forward_feature(obs.permute(0,2,3,1) * 0.5 + 0.5) 
        elif self.encoder_type == 'DINO':
            if isinstance(obs, tuple):
                visual_embedding = torch.stack([self.vision_encoder.forward_features(image) for image in obs], dim=1)
            else:
                visual_embedding = self.vision_encoder.forward_features(obs) 

        depth_embedding = None
        if self.with_depth:
            depth_obs = self.depth_resize(depth_obs.unsqueeze(1))
            depth_embedding = self.depth_encoder(depth_obs)

        visual_embedding_gripper = None
        depth_embedding_gripper  = None
        if self.with_gripper:
            visual_embedding_gripper = self.vision_encoder.forward_features(gripper_obs[0]) 
            gripper_depth_obs = self.depth_resize(gripper_obs[1].unsqueeze(1))
            depth_embedding_gripper = self.depth_encoder(gripper_depth_obs)

        tactile_embedding = None
        if self.with_tactile:
            tactile_obs = self.tactile_resize(tactile_obs)
            tactile_embedding = self.tactile_encoder(tactile_obs)

        # run sampling
        action_pred = self.conditional_sample(
            cond_data, 
            local_cond=ref_action,
            global_cond=(action_cond, \
            visual_embedding, \
            depth_embedding, 
            visual_embedding_gripper, depth_embedding_gripper, \
            tactile_embedding),
            hist_action=hist_action,
            lang=lang,
            proprio=proprio,
            **self.kwargs)
        

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]


        return action

    # ========= training  ============

    def compute_loss(self, trajectory, ref_action, action_cond, obs, depth_obs=None, gripper_obs=None, tactile_obs=None, lang=None, proprio=None, hist_action=None, decoupled_loss=False):
        batch_size = trajectory.shape[0]
        horizon = trajectory.shape[1]

        # Sample noise that we'll add to the trajectories
        if self.progressive_noise:
            noise = self.gen_progressive_noise(trajectory)
        else:
            noise = torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each action
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()

        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        if self.encoder_type == 'Theia':
            if isinstance(obs, tuple):
                visual_embedding = torch.stack([self.vision_encoder.forward_feature(image.permute(0,2,3,1) * 0.5 + 0.5) for image in obs], dim=1)
            else:
                visual_embedding = self.vision_encoder.forward_feature(obs.permute(0,2,3,1) * 0.5 + 0.5) 
        elif self.encoder_type == 'DINO':
            if isinstance(obs, tuple):
                visual_embedding = torch.stack([self.vision_encoder.forward_features(image) for image in obs], dim=1)
            else:
                visual_embedding = self.vision_encoder.forward_features(obs) 

        depth_embedding = None
        if self.with_depth:
            depth_obs = self.depth_resize(depth_obs.unsqueeze(1))
            depth_embedding = self.depth_encoder(depth_obs)

        visual_embedding_gripper = None
        depth_embedding_gripper  = None
        if self.with_gripper:
            visual_embedding_gripper = self.vision_encoder.forward_features(gripper_obs[0]) 
            gripper_depth_obs = self.depth_resize(gripper_obs[1].unsqueeze(1))
            depth_embedding_gripper = self.depth_encoder(gripper_depth_obs)

        tactile_embedding = None
        if self.with_tactile:
            tactile_obs = self.tactile_resize(tactile_obs)
            tactile_embedding = self.tactile_encoder(tactile_obs)

        
        cond_mask = None
        if self.with_cfg:
            cond_mask = (torch.rand(noisy_trajectory.shape[0], 1,  device = noisy_trajectory.device) > self.cond_drop_chance).float()

        # Predict the noise residual
        pred = self.model(noisy_trajectory, 
                          timesteps, 
                          cond=ref_action, 
                          context=action_cond,
                          visual_embedding=visual_embedding, 
                          depth_embedding=depth_embedding if self.with_depth else None,
                          gripper_embedding = (visual_embedding_gripper, depth_embedding_gripper) if self.with_gripper else None,
                          tactile_embedding = tactile_embedding if self.with_tactile else None,
                          lang=lang,
                          proprio=proprio,
                          hist_action=hist_action,
                          cond_mask=cond_mask)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        if decoupled_loss:
            target[..., -1] = torch.where(target[..., -1] > 0, 1, 0)
            gripper_loss = F.binary_cross_entropy_with_logits(pred[..., -1], target[..., -1], reduction='mean')
            ee_loss = F.mse_loss(pred[..., :-1], target[..., :-1], reduction='none')
            ee_loss = reduce(ee_loss, 'b ... -> b (...)', 'mean')
            ee_loss = ee_loss.mean()
            loss = ee_loss + 0.1 * gripper_loss
        else:
            loss = F.mse_loss(pred, target, reduction='none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()

            
        return loss


    def gen_progressive_noise(self, trajectory:torch.Tensor, alpha:int = 2):
        b, t, a = trajectory.shape
        
        eps_0 = torch.randn((b, 1, a), device=trajectory.device)
        all_noise = [eps_0]
        for i in range(1, t):
            eps_t = torch.normal(mean = torch.zeros((b,1,a)), std = torch.ones((b,1,a)) / (1 + alpha ** 2)).to(trajectory.device)
            eps_t = eps_t + alpha * all_noise[i-1] / torch.sqrt(torch.tensor(1 + alpha ** 2)).to(trajectory.device)
            all_noise.append(eps_t)

        all_noise = torch.cat(all_noise, dim = 1)

        return all_noise

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
