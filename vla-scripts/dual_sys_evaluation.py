import queue
import threading
import time
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from dataclasses import dataclass
from einops import rearrange

from typing import Optional
from transformers.generation.streamers import BaseStreamer


from calvin_agent.models.calvin_base_model import CalvinBaseModel


###################################
class ActionTokenTimingStreamer(BaseStreamer):
    """Streamer that records TTFT and TPOT during generation."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.device = device
        if self.device is not None and not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)
        self._should_sync = False
        if torch.cuda.is_available():
            if self.device is None:
                self._should_sync = True
            elif self.device.type == "cuda":
                self._should_sync = True
        self.reset()

    def _synchronize_device(self) -> None:
        if not self._should_sync:
            return

        if self.device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(self.device)

    def reset(self) -> None:
        self.start_time = None
        self.first_token_time = None
        self.end_time = None
        self.token_times = []
        self.token_count = 0
        self.finished = False
        self.prompt_token_count = 0
        self._prefill_handled = False

    def start(self) -> None:
        self._synchronize_device()
        self.start_time = time.perf_counter()

    def put(self, value) -> None:
        self._synchronize_device()
        now = time.perf_counter()
        num_tokens: int

        if isinstance(value, torch.Tensor):
            num_tokens = int(value.numel())
            is_prompt_chunk = value.ndim > 1 or (value.ndim == 1 and num_tokens > 1)
        else:
            try:
                num_tokens = len(value)
                is_prompt_chunk = num_tokens > 1
            except TypeError:
                num_tokens = 1
                is_prompt_chunk = False

        if not self._prefill_handled and is_prompt_chunk:
            # The first chunk delivered by the streamer contains the prompt tokens
            # that were fed into the model during prefill. We skip counting them so
            # TTFT reflects the latency until the first *generated* token.
            self._prefill_handled = True
            self.prompt_token_count = num_tokens
            return

        if not self._prefill_handled:
            # No prompt chunk was emitted; treat this as the first generated token.
            self._prefill_handled = True

        if self.first_token_time is None:
            self.first_token_time = now

        self.token_times.extend([now] * num_tokens)
        self.token_count += num_tokens

    def end(self) -> None:
        self._synchronize_device()
        self.end_time = time.perf_counter()
        self.finished = True

    def finalize(self) -> None:
        if not self.finished:
            self.end()

    def get_metrics(self):
        if self.start_time is None or self.first_token_time is None or self.token_count == 0:
            return None

        end_time = self.end_time if self.end_time is not None else self.token_times[-1]
        ttft = self.first_token_time - self.start_time

        if self.token_count > 1:
            tpot = (end_time - self.first_token_time) / (self.token_count - 1)
        else:
            tpot = 0.0

        total_time = end_time - self.start_time
        return {
            "ttft": ttft,
            "tpot": tpot,
            "token_count": self.token_count,
            "total_time": total_time,
        }
###################################


#add1
@dataclass
class TimingAggregator:
    """Keeps running statistics for latency measurements."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def update(self, duration: float) -> None:
        self.count += 1
        self.total += duration
        self.total_sq += duration * duration

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [float(self.count), float(self.total), float(self.total_sq)],
            device=device,
            dtype=torch.float64,
        )

    def merge(self, other: "TimingAggregator") -> None:
        self.count += other.count
        self.total += other.total
        self.total_sq += other.total_sq

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def std(self) -> float:
        if self.count == 0:
            return 0.0
        mean = self.mean()
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return math.sqrt(variance)

    def frequency(self) -> float:
        mean = self.mean()
        if mean == 0.0:
            return 0.0
        return 1.0 / mean
#add1_end

def get_openvla_prompt(instruction: str, tokenized_action: str = None) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


class DualSystemCalvinEvaluation(CalvinBaseModel):
    def __init__(self, model, processor, action_tokenizer):
        super().__init__()

        #self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        inferred_model = getattr(model, "module", model)
        try:
            inferred_param = next(inferred_model.parameters())
        except (AttributeError, StopIteration):
            try:
                inferred_param = next(model.parameters())
            except (AttributeError, StopIteration):
                inferred_param = None

        if inferred_param is not None:
            self.device = inferred_param.device
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



        self.processor = processor
        self.dual_sys = model
        self.dual_impl = getattr(self.dual_sys, "module", self.dual_sys)
        

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
        ########################
        self.ttft_records = []
        self.tpot_records = []
        ########################

        #add2
        self._generalist_stats = TimingAggregator()
        self._specialist_stats = TimingAggregator()
        self._control_stats = TimingAggregator()
        #add2_end

        # Threading primitives for asynchronous generalist execution
        self._generalist_queue: "queue.Queue[tuple[int, int, int, dict[str, torch.Tensor]]]" = queue.Queue(maxsize=1)
        self._generalist_ready_event = threading.Event()
        self._generalist_lock = threading.Lock()
        self._generalist_thread: Optional[threading.Thread] = None
        self._generalist_stop_event = threading.Event()
        self._pending_generalist = False
        self._generalist_request_counter = 0
        self._generalist_context = 0

        self._specialist_exec_counter = 0

        self._start_generalist_worker()

        
    def _start_generalist_worker(self) -> None:
        if self._generalist_thread is not None and self._generalist_thread.is_alive():
            return

        self._generalist_stop_event.clear()

        def _worker_loop():
            while not self._generalist_stop_event.is_set():
                try:
                    payload = self._generalist_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if payload is None:
                    self._generalist_queue.task_done()
                    break

                context_id, request_id, step_index, inputs = payload
                generalist_start = time.perf_counter()

                streamer = ActionTokenTimingStreamer(device=self.device)
                streamer.start()
                action, hidden_states = self.dual_impl.slow_system.predict_action(
                    streamer=streamer, do_sample=False, **inputs
                )
                streamer.finalize()
                timing_metrics = streamer.get_metrics()
                if timing_metrics is not None:
                    ttft = timing_metrics["ttft"]
                    tpot = timing_metrics["tpot"]
                    with self._generalist_lock:
                        self.ttft_records.append(ttft)
                        self.tpot_records.append(tpot)
                    print(
                        f"[Latency][System-2] Step {step_index}: TTFT={ttft:.4f}s, TPOT={tpot:.4f}s, Tokens={timing_metrics['token_count']}"
                    )

                action = torch.tensor(action).to(hidden_states.device).unsqueeze(0)
                action = rearrange(action, "b (f d) -> b f d", f=self.temporal_size)[:, :, :7]

                duration = time.perf_counter() - generalist_start
                with self._generalist_lock:
                    if context_id != self._generalist_context:
                        self._pending_generalist = False
                        self._generalist_queue.task_done()
                        self._generalist_ready_event.set()
                        continue

                    self.action = action
                    self.hidden_states = hidden_states
                    self._pending_generalist = False
                    self._generalist_stats.update(duration)
                self._generalist_ready_event.set()
                self._generalist_queue.task_done()

        self._generalist_thread = threading.Thread(target=_worker_loop, daemon=True)
        self._generalist_thread.start()

    def _stop_generalist_worker(self) -> None:
        if self._generalist_thread is None:
            return

        self._generalist_stop_event.set()
        try:
            self._generalist_queue.put_nowait(None)
        except queue.Full:
            pass
        self._generalist_thread.join(timeout=1.0)
        self._generalist_thread = None

    def _submit_generalist_request(self, inputs: dict[str, torch.Tensor], step: int, *, block: bool) -> bool:
        with self._generalist_lock:
            if self._pending_generalist:
                return False

            self._generalist_request_counter += 1
            request_id = self._generalist_request_counter
            context_id = self._generalist_context
            self._pending_generalist = True

        try:
            self._generalist_ready_event.clear()
            if block:
                self._generalist_queue.put((context_id, request_id, step, inputs))
                return True
            else:
                self._generalist_queue.put_nowait((context_id, request_id, step, inputs))
                return True
        except queue.Full:
            with self._generalist_lock:
                self._pending_generalist = False
            return False

    def _maybe_request_generalist(self, inputs: dict[str, torch.Tensor], step: int, wait: bool = False) -> None:
        success = self._submit_generalist_request(inputs, step, block=wait)
        if wait:
            self._generalist_ready_event.wait()
            self._generalist_ready_event.clear()
        elif not success:
            # Another generalist request is currently running; let it finish asynchronously.
            pass

    def reset(self,):
        """
        This is called
        """

        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)
        self.obs_buffer = None
        self.hist_action = []

        with self._generalist_lock:
            self._generalist_context += 1
            self.hidden_states = None
            self.action = None
            self._pending_generalist = False
        self._generalist_ready_event.clear()
        self._specialist_exec_counter = 0


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
        #gripper_image = self.processor.image_processor.apply_transform(Image.fromarray(gripper_image))[:3].unsqueeze(0).to(self.dual_sys.device)
        gripper_image = self.processor.image_processor.apply_transform(Image.fromarray(gripper_image))[:3].unsqueeze(0).to(self.device)

        tactile_image = None
        # tactile_image = torch.from_numpy(obs["rgb_obs"]['rgb_tactile']).permute(2,0,1).unsqueeze(0).to(self.dual_sys.device, dtype=torch.float) / 255
        depth_image = torch.from_numpy(obs["depth_obs"]['depth_static']).unsqueeze(0).to(self.device) - self.depth_min / (self.depth_max - self.depth_min)
        depth_gripper = torch.from_numpy(obs["depth_obs"]['depth_gripper']).unsqueeze(0).to(self.device) - self.gripper_depth_min / (self.gripper_depth_max - self.gripper_depth_min)

        prompt = get_openvla_prompt(instruction)
        inputs = self.processor(prompt, Image.fromarray(image)).to(self.device, dtype=torch.bfloat16)


        with self._generalist_lock:
            generalist_ready = self.hidden_states is not None

        if not generalist_ready:
            self._maybe_request_generalist(inputs, step, wait=True)
        elif (self._specialist_exec_counter + 1) % self.temporal_size == 0:
            self._maybe_request_generalist(inputs, step, wait=False)

        with self._generalist_lock:
            current_action = self.action
            current_hidden_states = self.hidden_states

        if current_action is None or current_hidden_states is None:
            raise RuntimeError("Generalist output was not ready for specialist inference.")


        remainder = (step + 1) % self.temporal_size
        num_cond_actions = self.temporal_size if remainder == 0 else self.temporal_size - remainder
        if step == 0:
            num_cond_actions = self.temporal_size

        zero_actions = torch.zeros(
            (1, self.temporal_size, 7), device=current_action.device, dtype=current_action.dtype
        )
        zero_actions[:, :num_cond_actions] = current_action[:, -num_cond_actions:]
        ref_actions = zero_actions

        state = torch.from_numpy(obs['robot_obs']).to(self.device, dtype=torch.float)
        state = torch.cat([state[:6], state[[-1]]], dim=-1).unsqueeze(0)

        if step == 0:
            self.obs_buffer = image

        prev_img = self.processor.image_processor.apply_transform(Image.fromarray(self.obs_buffer))[:3].unsqueeze(0).to(self.device)
        obs = (inputs["pixel_values"][:,:3].to(torch.float), prev_img)


        hist_action = torch.zeros((1,4,7)).to(self.device)
        available_hist_acts = len(self.hist_action)
        available_hist_acts = 4 if available_hist_acts > 4 else available_hist_acts
        if available_hist_acts > 0:
            hist_action[:, -available_hist_acts:] = torch.stack(self.hist_action[-available_hist_acts:], dim=0).unsqueeze(0).to(self.device)

        
        #add5
        specialist_start = time.perf_counter()
        #add5_end
        dp_action = self.dual_impl.ema_fast_system.ema_model.predict_action(
                                                            ref_action = ref_actions.to(torch.float),
                                                            action_cond = current_hidden_states.to(torch.float),
                                                            obs = obs,
                                                            depth_obs = depth_image,
                                                            gripper_obs = (gripper_image, depth_gripper),
                                                            tactile_obs = tactile_image,
                                                            lang= instruction,
                                                            proprio = state,
                                                            hist_action=hist_action,
                                                            )
        #add6
        self._specialist_stats.update(time.perf_counter() - specialist_start)
        #add6_end
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

        self._specialist_exec_counter += 1

        return action_prediction
    #add7
    def record_control_cycle(self, duration: float) -> None:
        self._control_stats.update(duration)

    def timing_summaries(self) -> dict:
        return {
            "generalist": self._generalist_stats,
            "specialist": self._specialist_stats,
            "control": self._control_stats,
        }
    #add7_end
#######################################################

    def report_latency_metrics(self) -> None:
        if not self.ttft_records:
            print("[Latency][System-2] No VLM inference calls were recorded.")
            return

        avg_ttft = sum(self.ttft_records) / len(self.ttft_records)
        avg_tpot = sum(self.tpot_records) / len(self.tpot_records) if self.tpot_records else 0.0
        print(
            f"[Latency][System-2] Average TTFT={avg_ttft:.4f}s, Average TPOT={avg_tpot:.4f}s over {len(self.ttft_records)} runs"
        )

    def get_latency_aggregates(self):
        return (
            float(sum(self.ttft_records)),
            float(len(self.ttft_records)),
            float(sum(self.tpot_records)),
            float(len(self.tpot_records)),
        )
    ###################################################

    def __del__(self):
        self._stop_generalist_worker()