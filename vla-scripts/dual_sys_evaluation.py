import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange
import time

from calvin_agent.models.calvin_base_model import CalvinBaseModel



def get_openvla_prompt(instruction: str, tokenized_action: str = None) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


class DualSystemCalvinEvaluation(CalvinBaseModel):
    def __init__(self, model, processor, action_tokenizer):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.processor = processor
        self.dual_sys = model
        self.action_tokenizer = action_tokenizer

        self.temporal_size = 8
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        self.action = None
        self.hidden_states = None
        self.obs_buffer = None

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]

        # Dataset statics (rougnly computed with 10k samples in CALVIN)
        self.depth_max = 6.2
        self.depth_min = 3.5
        self.gripper_depth_max = 2.0
        self.gripper_depth_min = 0

        self.hist_action = []

        
    def reset(self,):
        """
        This is called
        """

        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)
        self.obs_buffer = None
        self.hist_action = []


    def step(self, obs, instruction, step):
        """
        Args:
            obs: environment observations
            instruction: embedded language goal
        Returns:
            action: predicted action
        """

        image = obs["rgb_obs"]['rgb_static']
        gripper_image = obs["rgb_obs"]['rgb_gripper']
        gripper_image = self.processor.image_processor.apply_transform(Image.fromarray(gripper_image))[:3].unsqueeze(0).to(self.dual_sys.device)

        tactile_image = None
        # tactile_image = torch.from_numpy(obs["rgb_obs"]['rgb_tactile']).permute(2,0,1).unsqueeze(0).to(self.dual_sys.device, dtype=torch.float) / 255
        depth_image = torch.from_numpy(obs["depth_obs"]['depth_static']).unsqueeze(0).to(self.dual_sys.device) - self.depth_min / (self.depth_max - self.depth_min)
        depth_gripper = torch.from_numpy(obs["depth_obs"]['depth_gripper']).unsqueeze(0).to(self.dual_sys.device) - self.gripper_depth_min / (self.gripper_depth_max - self.gripper_depth_min)

        prompt = get_openvla_prompt(instruction)
        inputs = self.processor(prompt, Image.fromarray(image)).to(self.dual_sys.device, dtype=torch.bfloat16)


        if (step + 1) % 8 == 0 or step == 0: 
            # Run VLA Inference
            action, hidden_states = self.dual_sys.module.slow_system.predict_action(**inputs, do_sample=False)
            action = torch.tensor(action).to(hidden_states.device).unsqueeze(0)
            action = rearrange(action, 'b (f d) -> b f d', f=8)
            self.action = action[:,:,:7]
            self.hidden_states = hidden_states


        num_cond_actions = 8 - (step + 1) % 8
        if step == 0:
            num_cond_actions = 8

        zero_actions = torch.zeros((1, self.temporal_size, 7))
        zero_actions[:, :num_cond_actions] = self.action[:, -num_cond_actions:]
        ref_actions = zero_actions.to(self.action.device)

        state = torch.from_numpy(obs['robot_obs']).to(self.dual_sys.device, dtype=torch.float)
        state = torch.cat([state[:6], state[[-1]]], dim=-1).unsqueeze(0)

        if step == 0:
            self.obs_buffer = image

        prev_img = self.processor.image_processor.apply_transform(Image.fromarray(self.obs_buffer))[:3].unsqueeze(0).to(self.dual_sys.device)
        obs = (inputs["pixel_values"][:,:3].to(torch.float), prev_img)


        hist_action = torch.zeros((1,4,7)).to(self.dual_sys.device)
        available_hist_acts = len(self.hist_action)
        available_hist_acts = 4 if available_hist_acts > 4 else available_hist_acts
        if available_hist_acts > 0:
            hist_action[:, -available_hist_acts:] = torch.stack(self.hist_action[-available_hist_acts:], dim=0).unsqueeze(0).to(self.dual_sys.device)


        dp_action = self.dual_sys.module.ema_fast_system.ema_model.predict_action(
                                                            ref_action = ref_actions.to(torch.float),
                                                            action_cond = self.hidden_states.to(torch.float),
                                                            obs = obs,
                                                            depth_obs = depth_image,
                                                            gripper_obs = (gripper_image, depth_gripper),
                                                            tactile_obs = tactile_image,
                                                            lang= instruction,
                                                            proprio = state,
                                                            hist_action=hist_action,
                                                            )
        self.obs_buffer = image
        action = np.array(dp_action.tolist())


        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = action  
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

        # Ensemble temporally to predict action
        action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)


        if action_prediction[-1] < -0.5:
            action_prediction[-1] = -1
        else:
            action_prediction[-1] = 1

        self.hist_action.append(torch.from_numpy(action_prediction))

        return action_prediction