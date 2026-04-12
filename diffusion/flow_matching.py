from copy import deepcopy

import torch
import torch as th

from diffusion.losses import geodesic_distance
from diffusion.nn import sum_flat
from utils.rotation_conversions import rotation_6d_to_matrix_safe


class FlowMatching:
    def __init__(
        self,
        *,
        num_timesteps,
        sigma_min=1e-4,
        solver="euler",
        timestep_scale=1000.0,
        sampling_steps=None,
        lambda_fs=0.0,
        lambda_geo=0.0,
        lambda_confidence_recon=0.0,
        lambda_repair_recon=0.0,
        lambda_root=0.0,
        lambda_velocity=0.0,
    ):
        self.num_timesteps = int(num_timesteps)
        self.sigma_min = float(sigma_min)
        self.solver = solver
        self.timestep_scale = float(timestep_scale)
        self.sampling_steps = int(sampling_steps or num_timesteps)
        self.lambda_fs = lambda_fs
        self.lambda_geo = lambda_geo
        self.lambda_confidence_recon = lambda_confidence_recon
        self.lambda_repair_recon = lambda_repair_recon
        self.lambda_root = lambda_root
        self.lambda_velocity = lambda_velocity
        self.l2_loss = lambda a, b: (a - b) ** 2

    def _append_time_dims(self, t, x):
        while t.dim() < x.dim():
            t = t.unsqueeze(-1)
        return t.to(device=x.device, dtype=x.dtype)

    def _scale_timesteps(self, t):
        return t.float() * self.timestep_scale

    def interpolate(self, x_start, t, noise):
        t = self._append_time_dims(t, x_start)
        return (1.0 - t) * x_start + t * noise

    def velocity_target(self, x_start, noise):
        return noise - x_start

    def predict_x0(self, x_t, t, v_theta):
        return x_t - self._append_time_dims(t, x_t) * v_theta

    def _predict_velocity_from_x0(self, x_t, t, x0):
        t = self._append_time_dims(t, x_t).clamp(min=self.sigma_min)
        return (x_t - x0) / t

    def temporal_spatial_masked_l2(self, a, b, temp_mask, spat_mask, lengths, n_joints):
        loss = self.l2_loss(a, b)
        temp_masked_loss = loss * temp_mask.float()
        spat_temp_masked_loss = temp_masked_loss * spat_mask.float().transpose(1, 3)
        loss = sum_flat(spat_temp_masked_loss)
        non_zero_elements = lengths * n_joints * a.size(2)
        return loss / (non_zero_elements + 1e-8)

    def weighted_temporal_spatial_l2(self, a, b, temp_mask, spat_mask, weights):
        if weights.dim() == 3:
            weights = weights.unsqueeze(2)
        combined_mask = temp_mask.float() * spat_mask.float().transpose(1, 3) * weights.float()
        loss = sum_flat(self.l2_loss(a, b) * combined_mask)
        denom = sum_flat(combined_mask) * a.size(2)
        return loss / (denom + 1e-8)

    def weighted_feature_l2(self, a, b, weights):
        if weights.dim() == 3:
            weights = weights.unsqueeze(2)
        weighted_loss = self.l2_loss(a, b) * weights.float()
        loss = sum_flat(weighted_loss)
        denom = sum_flat(weights.float()) * a.size(2)
        return loss / (denom + 1e-8)

    def confidence_weights(self, confidence):
        if confidence is None:
            return None
        return confidence.clamp(0.0, 1.0)

    def apply_reference_fusion(self, prediction, reference_motion, confidence):
        if reference_motion is None or confidence is None:
            return prediction
        reference_motion = reference_motion.to(prediction.device, dtype=prediction.dtype)
        confidence = confidence.to(prediction.device, dtype=prediction.dtype)
        reliability = self.confidence_weights(confidence)
        return torch.lerp(prediction, reference_motion, reliability)

    def get_reference_fusion_inputs(self, model_kwargs):
        if not model_kwargs:
            return None, None
        conditioning = model_kwargs.get("y")
        if conditioning is None:
            return None, None
        return conditioning.get("reference_motion"), conditioning.get("soft_confidence_mask")

    def geodesic_loss(self, a, b, temp_mask, spat_mask, lengths, n_joints):
        rots_target = rotation_6d_to_matrix_safe(a.permute(0, 3, 1, 2)[..., 3:9])
        rots_pred = rotation_6d_to_matrix_safe(b.permute(0, 3, 1, 2)[..., 3:9])
        loss = geodesic_distance(rots_pred, rots_target).permute(0, 2, 3, 1)
        temp_masked_loss = loss * temp_mask.float()
        spat_temp_masked_loss = temp_masked_loss * spat_mask.float().transpose(1, 3)
        loss = sum_flat(spat_temp_masked_loss)
        non_zero_elements = lengths * n_joints
        return loss / (non_zero_elements + 1e-8)

    def foot_sliding_loss(self, a, b, temp_mask, relative=True):
        fc = a[..., 12, :-1] * temp_mask[..., 0, 1:] != 0
        vel_tgt = a[..., :3, 1:] - a[..., :3, :-1]
        vel_pred = b[..., :3, 1:] - b[..., :3, :-1]
        if relative:
            loss_term = self.l2_loss(vel_tgt, vel_pred).sum(dim=-2).sqrt() * fc
        else:
            loss_term = (vel_pred ** 2).sum(dim=-2).sqrt() * fc
        loss = sum_flat(loss_term)
        non_zero_elements = 1 + torch.count_nonzero(fc, dim=(-1, -2))
        return loss / non_zero_elements

    def _model_velocity(self, model, x_t, t, model_kwargs=None, get_layer_activation=-1):
        if model_kwargs is None:
            model_kwargs = {}
        if 0 <= get_layer_activation < getattr(model, "num_layers", -1):
            model_output, activations = model(
                x_t,
                self._scale_timesteps(t),
                get_layer_activation=get_layer_activation,
                **model_kwargs,
            )
            return model_output, activations
        return model(x_t, self._scale_timesteps(t), **model_kwargs), None

    def _apply_x0_constraints(self, pred_x0, model_kwargs=None):
        conditioning = model_kwargs.get("y") if model_kwargs is not None else None
        if conditioning is not None and "inpainting_mask" in conditioning and "inpainted_motion" in conditioning:
            inpainting_mask = conditioning["inpainting_mask"]
            inpainted_motion = conditioning["inpainted_motion"]
            pred_x0 = (pred_x0 * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
        reference_motion, confidence = self.get_reference_fusion_inputs(model_kwargs)
        return self.apply_reference_fusion(pred_x0, reference_motion, confidence)

    def _guided_prediction(
        self,
        model,
        x_t,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        get_layer_activation=-1,
    ):
        velocity, activations = self._model_velocity(
            model,
            x_t,
            t,
            model_kwargs=model_kwargs,
            get_layer_activation=get_layer_activation,
        )
        pred_x0 = self.predict_x0(x_t, t, velocity)
        pred_x0 = self._apply_x0_constraints(pred_x0, model_kwargs=model_kwargs)
        if denoised_fn is not None:
            pred_x0 = denoised_fn(pred_x0)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)
        guided_velocity = self._predict_velocity_from_x0(x_t, t, pred_x0)
        return guided_velocity, pred_x0, activations

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        mask = model_kwargs["y"]["lengths_mask"]
        lengths = model_kwargs["y"]["lengths"]
        actual_joints = model_kwargs["y"]["n_joints"]
        joints_mask = model_kwargs["y"]["joints_mask"][:, :, :, 1, 1:]
        mean = model_kwargs["y"]["mean"][..., None]
        std = model_kwargs["y"]["std"][..., None]
        confidence = model_kwargs["y"].get("soft_confidence_mask")
        reference_motion = model_kwargs["y"].get("reference_motion")
        if confidence is None:
            confidence = torch.ones_like(x_start[:, :, :1, :])
        confidence = confidence.clamp(0.0, 1.0)
        reference_reliability = self.confidence_weights(confidence)
        if reference_motion is None:
            reference_motion = x_start
        else:
            reference_motion = reference_motion.to(x_start.device, dtype=x_start.dtype)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        x_t = self.interpolate(x_start, t, noise)
        velocity_target = self.velocity_target(x_start, noise)
        velocity_pred = model(x_t, self._scale_timesteps(t), **model_kwargs)
        assert velocity_pred.shape == velocity_target.shape == x_start.shape

        terms = {}
        terms["l_simple"] = self.temporal_spatial_masked_l2(
            velocity_target,
            velocity_pred,
            mask,
            joints_mask,
            lengths,
            actual_joints,
        )
        terms["velocity_matching_loss"] = terms["l_simple"]
        terms["loss"] = terms["l_simple"].clone()

        pred_x0 = self.predict_x0(x_t, t, velocity_pred)
        if self.lambda_confidence_recon > 0.0:
            reliable_weights = reference_reliability.clamp(min=0.0, max=1.0)
            terms["confidence_recon_loss"] = self.weighted_temporal_spatial_l2(
                reference_motion,
                pred_x0,
                mask,
                joints_mask,
                reliable_weights,
            )
            terms["loss"] = terms["loss"] + self.lambda_confidence_recon * terms["confidence_recon_loss"]
        if self.lambda_repair_recon > 0.0:
            repair_weights = (1.0 - reference_reliability).clamp(min=0.0, max=1.0)
            terms["repair_recon_loss"] = self.weighted_temporal_spatial_l2(
                x_start,
                pred_x0,
                mask,
                joints_mask,
                repair_weights,
            )
            terms["loss"] = terms["loss"] + self.lambda_repair_recon * terms["repair_recon_loss"]

        x_start_denorm = (x_start * std) + mean
        pred_x0_denorm = (pred_x0 * std) + mean

        if self.lambda_root > 0.0:
            root_weights = mask[:, :1, :, :] * (1.0 + (1.0 - reference_reliability[:, :1]))
            root_target = torch.cat([x_start_denorm[:, :1, :3, :], x_start_denorm[:, :1, 9:12, :]], dim=2)
            root_output = torch.cat([pred_x0_denorm[:, :1, :3, :], pred_x0_denorm[:, :1, 9:12, :]], dim=2)
            terms["root_consistency_loss"] = self.weighted_feature_l2(root_target, root_output, root_weights)
            terms["loss"] = terms["loss"] + self.lambda_root * terms["root_consistency_loss"]
        if self.lambda_velocity > 0.0:
            velocity_weights = 1.0 + (1.0 - reference_reliability)
            terms["velocity_consistency_loss"] = self.weighted_temporal_spatial_l2(
                x_start_denorm[:, :, 9:12, :],
                pred_x0_denorm[:, :, 9:12, :],
                mask,
                joints_mask,
                velocity_weights,
            )
            terms["loss"] = terms["loss"] + self.lambda_velocity * terms["velocity_consistency_loss"]
        if self.lambda_geo > 0.0:
            terms["geodesic_loss"] = self.geodesic_loss(
                x_start_denorm,
                pred_x0_denorm,
                mask,
                joints_mask,
                lengths,
                actual_joints,
            )
            terms["loss"] = terms["loss"] + self.lambda_geo * terms["geodesic_loss"]
        if self.lambda_fs > 0.0:
            terms["foot_sliding_loss"] = self.foot_sliding_loss(x_start_denorm, pred_x0_denorm, mask, relative=True)
            terms["loss"] = terms["loss"] + self.lambda_fs * terms["foot_sliding_loss"]

        return terms

    def _time_grid(self, device, dtype, skip_timesteps=0):
        step_count = max(int(self.sampling_steps), 1)
        time_grid = torch.linspace(1.0, self.sigma_min, steps=step_count + 1, device=device, dtype=dtype)
        start_index = min(max(int(skip_timesteps), 0), step_count - 1)
        return time_grid[start_index:]

    def _initialize_state(self, shape, device, noise=None, init_image=None, start_t=1.0, const_noise=False):
        if noise is None:
            noise = th.randn(*shape, device=device)
        else:
            noise = noise.to(device)
        if const_noise:
            noise = noise[[0]].repeat(shape[0], 1, 1, 1)
        img = noise
        if init_image is not None:
            init_image = init_image.to(device=device, dtype=img.dtype)
            t = torch.full((shape[0],), float(start_t), device=device, dtype=img.dtype)
            img = self.interpolate(init_image, t, img)
        return img

    def _solver_step(self, model, x_t, current_t, next_t, clip_denoised, denoised_fn, model_kwargs):
        batch_t = torch.full((x_t.shape[0],), float(current_t), device=x_t.device, dtype=x_t.dtype)
        dt = float(next_t - current_t)

        if self.solver == "euler":
            velocity, pred_x0, _ = self._guided_prediction(
                model,
                x_t,
                batch_t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            return x_t + dt * velocity, pred_x0

        if self.solver == "midpoint":
            k1, _, _ = self._guided_prediction(
                model,
                x_t,
                batch_t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            mid_t = current_t + 0.5 * dt
            mid_batch_t = torch.full((x_t.shape[0],), float(mid_t), device=x_t.device, dtype=x_t.dtype)
            mid_state = x_t + 0.5 * dt * k1
            k2, pred_x0, _ = self._guided_prediction(
                model,
                mid_state,
                mid_batch_t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            return x_t + dt * k2, pred_x0

        if self.solver == "rk4":
            k1, _, _ = self._guided_prediction(
                model,
                x_t,
                batch_t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            mid_t = current_t + 0.5 * dt
            mid_batch_t = torch.full((x_t.shape[0],), float(mid_t), device=x_t.device, dtype=x_t.dtype)
            k2, _, _ = self._guided_prediction(
                model,
                x_t + 0.5 * dt * k1,
                mid_batch_t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            k3, _, _ = self._guided_prediction(
                model,
                x_t + 0.5 * dt * k2,
                mid_batch_t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            next_batch_t = torch.full((x_t.shape[0],), float(next_t), device=x_t.device, dtype=x_t.dtype)
            k4, pred_x0, _ = self._guided_prediction(
                model,
                x_t + dt * k3,
                next_batch_t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            x_next = x_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return x_next, pred_x0

        raise NotImplementedError(f"Unknown flow matching solver: {self.solver}")

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
        get_activations=None,
    ):
        if cond_fn is not None or cond_fn_with_grad:
            raise NotImplementedError("FlowMatching does not implement cond_fn-guided sampling.")
        if randomize_class:
            raise NotImplementedError("FlowMatching does not implement class randomization.")
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        time_grid = self._time_grid(device=device, dtype=torch.float32, skip_timesteps=skip_timesteps)
        img = self._initialize_state(
            shape,
            device=device,
            noise=noise,
            init_image=init_image,
            start_t=float(time_grid[0].item()),
            const_noise=const_noise,
        )
        time_pairs = list(zip(time_grid[:-1], time_grid[1:]))
        if progress:
            from tqdm.auto import tqdm

            time_pairs = tqdm(time_pairs)

        for current_t, next_t in time_pairs:
            with th.no_grad():
                img, pred_x0 = self._solver_step(
                    model,
                    img,
                    float(current_t.item()),
                    float(next_t.item()),
                    clip_denoised,
                    denoised_fn,
                    model_kwargs,
                )
            yield {"sample": img.detach(), "pred_xstart": pred_x0.detach()}

        final_t = torch.full((shape[0],), float(time_grid[-1].item()), device=device, dtype=img.dtype)
        with th.no_grad():
            _, pred_x0, _ = self._guided_prediction(
                model,
                img,
                final_t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
        yield {"sample": pred_x0.detach(), "pred_xstart": pred_x0.detach()}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        get_activations=None,
    ):
        final = None
        dump = [] if dump_steps is not None else None
        for i, sample in enumerate(
            self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                skip_timesteps=skip_timesteps,
                init_image=init_image,
                randomize_class=randomize_class,
                cond_fn_with_grad=cond_fn_with_grad,
                const_noise=const_noise,
                get_activations=get_activations,
            )
        ):
            final = sample
            if dump is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
        if dump is not None:
            return dump
        return final["sample"]

    def p_sample_single_timestep(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        init_image=None,
        randomize_class=False,
        const_noise=False,
        get_activations=None,
    ):
        if cond_fn is not None or randomize_class:
            raise NotImplementedError("FlowMatching single-step sampling supports only unconditional model evaluation.")
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        get_activations = get_activations or {"layer": -1, "timestep": self.num_timesteps - 1}
        timestep = int(get_activations.get("timestep", self.num_timesteps - 1))
        layer = int(get_activations.get("layer", -1))
        timestep = max(0, min(timestep, self.num_timesteps - 1))
        t_value = max(self.sigma_min, float(timestep) / max(self.num_timesteps - 1, 1))

        img = self._initialize_state(
            shape,
            device=device,
            noise=noise,
            init_image=init_image,
            start_t=t_value,
            const_noise=const_noise,
        )
        t = torch.full((shape[0],), t_value, device=device, dtype=img.dtype)
        with th.no_grad():
            _, pred_x0, activations = self._guided_prediction(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                get_layer_activation=layer,
            )
        activations_dict = {timestep: activations} if activations is not None else {}
        return pred_x0.detach(), activations_dict

    def _sample_with_solver(self, solver, sample_fn, *args, **kwargs):
        original_solver = self.solver
        self.solver = solver
        try:
            return sample_fn(*args, **kwargs)
        finally:
            self.solver = original_solver

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=False, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0, skip_timesteps=0, init_image=None, randomize_class=False, cond_fn_with_grad=False, dump_steps=None, const_noise=False):
        return self._sample_with_solver(
            "midpoint",
            self.p_sample_loop,
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            dump_steps=dump_steps,
            const_noise=const_noise,
        )

    def plms_sample_loop(self, model, shape, noise=None, clip_denoised=False, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, skip_timesteps=0, init_image=None, randomize_class=False, cond_fn_with_grad=False, dump_steps=None, const_noise=False, order=2):
        return self._sample_with_solver(
            "rk4",
            self.p_sample_loop,
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            dump_steps=dump_steps,
            const_noise=const_noise,
        )