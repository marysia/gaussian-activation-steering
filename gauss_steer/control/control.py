import dataclasses
import typing
import warnings

import torch
from transformers import PretrainedConfig, PreTrainedModel

if typing.TYPE_CHECKING:
    from gauss_steer.control.extract import ControlVector


class ControlModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(self, model: PreTrainedModel, layer_ids: typing.Iterable[int]):
        """
        **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

        Build a new ControlModel around a model instance, initializing control on
        the layers specified in `layer_ids`.
        """

        super().__init__()
        self.model = model

        layers = model_layer_list(model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]
        print("ControlModel wrapping layers:", self.layer_ids)
        for layer_id in layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, ControlModule):
                layers[layer_id] = ControlModule(layer)
            else:
                warnings.warn(
                    "Trying to rewrap a wrapped model! Probably not what you want! Try calling .unwrap first."
                )

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device

    def unwrap(self) -> PreTrainedModel:
        """
        Removes the mutations done to the wrapped model and returns it.
        After using this method, `set_control` and `reset` will not work.
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layers[layer_id] = layers[layer_id].block
        return self.model

    def reset(self) -> None:
        """
        Resets the control for all layer_ids, returning the model to base behavior.
        """
        self.set_raw_control(None)

    def set_control(
        self,
        control: "ControlVector",
        coeff_function: typing.Callable[[int, int], float],
        **kwargs,
    ) -> None:
        """
        Set a `ControlVector` with coefficients determined by a function.

        Args:
            control: The ControlVector to apply.
            coeff_function: A function that takes (layer_index, num_layers) and returns a float coefficient.
            **kwargs: Additional arguments for set_raw_control.
        """
        raw_control = {}
        num_layers = len(self.layer_ids)
        for i, layer_id in enumerate(self.layer_ids):
            coeff = coeff_function(i, num_layers)
            raw_control[layer_id] = torch.tensor(
                coeff * control.directions[layer_id]
            ).to(self.model.device, dtype=self.model.dtype)
        self.set_raw_control(raw_control, **kwargs)

    def set_raw_control(
        self, control: dict[int, torch.Tensor] | None, **kwargs
    ) -> None:
        """
        Set or remove control parameters to the layers this ControlModel handles.
        The keys of `control` should be equal to or a superset of the `layer_ids` passed to __init__.
        Only those layers will be controlled, any others in `control` will be ignored.

        Passing `control=None` will reset the control tensor for all layer_ids, making the model act
        like a non-control model.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer: ControlModule = layers[layer_id]  # type: ignore
            if control is None:
                layer.reset()
            else:
                layer.set_control(BlockControlParams(control[layer_id], **kwargs))

    def set_control_with_linear_schedule(
        self,
        control: "ControlVector",
        start_coeff: float,
        end_coeff: float,
        **kwargs,
    ) -> None:
        """
        Set a `ControlVector` with a linear schedule for coefficients.

        Args:
            control: The ControlVector to apply.
            start_coeff: The coefficient for the first controlled layer.
            end_coeff: The coefficient for the last controlled layer.
            **kwargs: Additional arguments for set_raw_control.
        """

        def linear_fn(i, n):
            if n == 1:
                return start_coeff
            return start_coeff + (end_coeff - start_coeff) * i / (n - 1)

        self.set_control(control, linear_fn, **kwargs)

    def set_control_with_parabolic_schedule(
        self,
        control: "ControlVector",
        max_coeff: float,
        min_coeff: float,
        **kwargs,
    ) -> None:
        """
        Set a `ControlVector` with a parabolic schedule for coefficients,
        peaking at the middle layer.

        Args:
            control: The ControlVector to apply.
            max_coeff: The maximum coefficient (at the peak of the parabola).
            min_coeff: The minimum coefficient (at the edges).
            **kwargs: Additional arguments for set_raw_control.
        """

        def parabolic_fn(i, n):
            if n == 1:
                return max_coeff
            # Shift and scale parabola to be between 0 and 1
            x = i / (n - 1)
            parabola = -4 * (x - 0.5) ** 2 + 1
            return min_coeff + (max_coeff - min_coeff) * parabola

        self.set_control(control, parabolic_fn, **kwargs)

    def set_control_with_gaussian_schedule(
        self,
        control: "ControlVector",
        peak_coeff: float,
        std_dev_factor: float = 0.25,
        **kwargs,
    ) -> None:
        """
        Set a `ControlVector` with a Gaussian schedule for coefficients,
        peaking at the middle layer.

        Args:
            control: The ControlVector to apply.
            peak_coeff: The maximum coefficient (at the peak of the Gaussian).
            std_dev_factor: The standard deviation of the Gaussian as a factor of the
                            number of controlled layers. Controls the "width" of the peak.
                            (default: 0.25)
            **kwargs: Additional arguments for set_raw_control.
        """
        import numpy as np

        def gaussian_fn(i, n):
            if n == 1:
                return peak_coeff

            # Define Gaussian over 1-based positions p = i+1 with center at floor(n/2)
            # so that the peak aligns with 1-based layer floor(n/2) (0-based i = floor(n/2)-1).
            p = i + 1
            mu = n // 2  # 1-based center position (lower middle)
            std_dev = (n - 1) * std_dev_factor
            if std_dev == 0:
                center_i = max(0, min(n - 1, mu - 1))
                return peak_coeff if i == center_i else 0.0

            # Use the Gaussian function to determine the coefficient
            # The peak of the Gaussian will be 1 (at p == mu), scaled by peak_coeff
            gaussian_val = np.exp(-0.5 * ((p - mu) / std_dev) ** 2)

            return peak_coeff * gaussian_val

        self.set_control(control, gaussian_fn, **kwargs)

    def set_gaussian_random(
        self,
        control: "ControlVector",
        peak_coeff: float,
        std_dev_factor: float = 0.25,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        """
        Randomly assign Gaussian-schedule strengths to layers while preserving total energy.

        Computes the same Gaussian-shaped coefficients as `set_control_with_gaussian_schedule`
        (with the same `peak_coeff` and `std_dev_factor`), then randomly permutes
        these values across the controlled layers. This keeps the multiset and thus the
        total L1 energy identical to the Gaussian schedule.

        Args:
            control: The ControlVector to apply.
            peak_coeff: The maximum coefficient at the peak of the Gaussian curve.
            std_dev_factor: Standard deviation as a fraction of the controlled depth.
            seed: Optional RNG seed for reproducibility of the permutation.
            **kwargs: Additional arguments for set_raw_control.
        """
        import numpy as np

        n = len(self.layer_ids)
        if n == 0:
            return self.set_raw_control(None, **kwargs)

        # Build base Gaussian coefficients (same semantics as in set_control_with_gaussian_schedule)
        if n == 1:
            coeffs = np.array([peak_coeff], dtype=float)
        else:
            mu = n // 2  # 1-based center
            std_dev = (n - 1) * float(std_dev_factor)
            if std_dev == 0:
                coeffs = np.zeros(n, dtype=float)
                center_i = max(0, min(n - 1, mu - 1))
                coeffs[center_i] = float(peak_coeff)
            else:
                pos = np.arange(1, n + 1, dtype=float)  # 1..n
                gaussian_vals = np.exp(-0.5 * ((pos - mu) / std_dev) ** 2)
                coeffs = peak_coeff * gaussian_vals

        # Randomly permute coefficients
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        coeffs = coeffs[perm]

        def fn(i: int, _n: int) -> float:
            return float(coeffs[i])

        self.set_control(control, fn, **kwargs)

    def set_uniform_gaussian(
        self,
        control: "ControlVector",
        peak_coeff: float,
        std_dev_factor: float = 0.25,
        **kwargs,
    ) -> None:
        """
        Uniform strengths with the same total energy as the Gaussian schedule.

        We compute the Gaussian-shaped coefficients as in `set_control_with_gaussian_schedule`
        and then distribute their total sum uniformly across all controlled layers. This keeps
        the L1 sum of coefficients identical to the Gaussian schedule, but makes them constant
        per layer ("constant energy").

        Args:
            control: The ControlVector to apply.
            peak_coeff: The maximum coefficient at the peak of the Gaussian curve.
            std_dev_factor: Standard deviation as a fraction of the controlled depth.
            **kwargs: Additional arguments for set_raw_control.
        """
        import numpy as np

        n = len(self.layer_ids)
        if n == 0:
            return self.set_raw_control(None, **kwargs)

        # Base Gaussian coefficients
        if n == 1:
            total = float(peak_coeff)
        else:
            mu = n // 2  # 1-based center
            std_dev = (n - 1) * float(std_dev_factor)
            if std_dev == 0:
                total = float(peak_coeff)
            else:
                pos = np.arange(1, n + 1, dtype=float)
                gaussian_vals = np.exp(-0.5 * ((pos - mu) / std_dev) ** 2)
                total = float(peak_coeff) * float(gaussian_vals.sum())

        # Uniform distribution preserving total L1 energy
        uniform_coeff = total / float(n)

        def fn(i: int, _n: int) -> float:
            return float(uniform_coeff)

        self.set_control(control, fn, **kwargs)

    def set_box_filter(
        self,
        control: "ControlVector",
        peak_coeff: float,
        std_dev_factor: float = 0.25,
        constant: float = 0.0,
        use_fwhm: bool = False,
        k_layers: int | None = None,
        **kwargs,
    ) -> None:
        """
        Rectangular ('boxcar') schedule centered at the Gaussian peak, with the
        SAME total L1 energy as the Gaussian schedule.

        By default we choose the box width to match the Gaussian FWHM
        (FWHM ≈ 2.3548 * sigma), where sigma = std_dev_factor * (n-1).
        Alternatively, set k_layers to specify the exact number of layers in the box.

        Args:
            control: The ControlVector to apply.
            peak_coeff: Maximum coefficient at the Gaussian peak (used only to compute
                        the Gaussian's total L1 energy that we preserve).
            std_dev_factor: Sigma as a fraction of depth, like in your Gaussian code.
            use_fwhm: If True, set box width from Gaussian FWHM; else fall back to
                    a width of max(1, round(2 * sigma)) layers.
            k_layers: If provided, use exactly this many layers in the box (overrides FWHM).
            **kwargs: Passed to set_raw_control via set_control.

        Behavior:
            - Compute the Gaussian coefficients exactly as in your Gaussian schedule
            to get the total L1 energy 'total'.
            - Choose a centered index band [start, end) (size = band_size).
            - Set every layer in the band to coeff = total / band_size; others to 0.
            - Call self.set_control(control, fn, **kwargs).
        """
        import numpy as np

        n = len(self.layer_ids)
        if n == 0:
            return self.set_raw_control(None, **kwargs)

        # ----- 1) Compute Gaussian total L1 energy (mirror of your uniform helper) -----
        if n == 1:
            total = float(peak_coeff)
            std_dev = 0.0
        else:
            std_dev = (n - 1) * float(std_dev_factor)
            if std_dev == 0:
                total = float(peak_coeff)
            else:
                pos = np.arange(1, n + 1, dtype=float)  # 1..n positions
                mu = n // 2  # lower middle in 1-based
                gaussian_vals = np.exp(-0.5 * ((pos - mu) / std_dev) ** 2)
                total = float(peak_coeff) * float(gaussian_vals.sum())

        # ----- 2) Decide box width (in layers) -----
        # Center of the box matches the Gaussian center (lower middle in 1-based)
        center = max(0, min(n - 1, (n // 2) - 1))

        if k_layers is not None:
            band_size = int(max(1, min(n, k_layers)))
        else:
            if std_dev == 0:
                band_size = 1
            else:
                if use_fwhm:
                    # FWHM ≈ 2.354820045... * sigma
                    fwhm = 2.354820045 * std_dev
                    band_size = int(max(1, min(n, round(fwhm))))
                else:
                    # Reasonable fallback: full width ≈ 2*sigma
                    band_size = int(max(1, min(n, round(2.0 * std_dev))))

        # ensure odd band (centered) if possible
        if band_size % 2 == 0:
            band_size = min(n, band_size + 1)

        half = band_size // 2
        start = max(0, center - half)
        end = min(n, start + band_size)
        # if we clipped at the left or right edge, re-center the band
        start = max(0, min(start, n - band_size))
        end = start + band_size

        # ----- 3) Allocate uniform coeff inside band so sum |coeffs| == total -----
        if band_size <= 0:
            # Fallback: no band -> zero schedule
            def fn(_i: int, _n: int) -> float:
                return 0.0

        else:
            if constant != 0.0:
                coeff = constant
            else:
                coeff = total / float(band_size)

            def fn(i: int, _n: int) -> float:
                return float(coeff) if (start <= i < end) else 0.0

        # ----- 4) Apply -----
        self.set_control(control, fn, **kwargs)

    def set_box_filter_on_given_starting_layer_and_streach_to_l1_gauss_energy(
        self,
        control: "ControlVector",
        peak_coeff: float,
        std_dev_factor: float = 0.25,
        start_layer: int = 1,
        k_layers: int | None = None,
        **kwargs,
    ) -> None:
        """
        Box filter centered at a given layer (1-based), stretched to match Gaussian L1 energy.

        Semantics:
        - start_layer is a 1-based position in the controlled order (same as
          set_box_filter_on_given_starting_layer).
        - The box is centered at that position; width is either k_layers (if provided)
          or derived from the Gaussian width (FWHM by default). The width is forced odd
          to keep an exact center when possible.
        - Coefficients inside the band are constant so that the total L1 energy matches
          the Gaussian schedule computed with (peak_coeff, std_dev_factor).
        """

        import numpy as np

        n = len(self.layer_ids)
        if n == 0:
            return self.set_raw_control(None, **kwargs)

        # 1) Compute Gaussian total L1 energy (mirror of uniform Gaussian helper)
        if n == 1:
            total = float(peak_coeff)
            std_dev = 0.0
        else:
            std_dev = (n - 1) * float(std_dev_factor)
            if std_dev == 0:
                total = float(peak_coeff)
            else:
                pos = np.arange(1, n + 1, dtype=float)  # 1..n positions
                mu = n // 2  # lower middle in 1-based
                gaussian_vals = np.exp(-0.5 * ((pos - mu) / std_dev) ** 2)
                total = float(peak_coeff) * float(gaussian_vals.sum())

        # 2) Decide box width (in layers)
        if k_layers is not None:
            band_size = int(max(1, min(n, k_layers)))
        else:
            if std_dev == 0:
                band_size = 1
            else:
                # FWHM ≈ 2.354820045... * sigma
                fwhm = 2.354820045 * std_dev
                band_size = int(max(1, min(n, round(fwhm))))

        # Ensure odd band (for clean centering) if possible
        if band_size % 2 == 0:
            band_size = min(n, band_size + 1)

        half = band_size // 2

        # Interpret start_layer as 1-based CENTER -> convert to 0-based index
        center_idx = int(start_layer) - 1
        center_idx = max(0, min(n - 1, center_idx))

        start = max(0, center_idx - half)
        end = min(n, start + band_size)
        # if we clipped at an edge, re-center within bounds
        start = max(0, min(start, n - band_size))
        end = start + band_size

        # 3) Allocate uniform coeff inside band so sum |coeffs| == total
        if band_size <= 0:

            def fn(_i: int, _n: int) -> float:
                return 0.0

        else:
            coeff = total / float(band_size)

            print(f"Box filter on given starting layer {start_layer}:")
            print(f"  Band size: {band_size}")
            print(f"  Coefficients: {coeff}")
            print(f"total l1 energy: {coeff*band_size} vs gaussian total: {total}")

            def fn(i: int, _n: int) -> float:
                return float(coeff) if (start <= i < end) else 0.0

        # 4) Apply
        self.set_control(control, fn, **kwargs)

    def set_harp_bump_gaussian(
        self,
        control: "ControlVector",
        peak_coeff: float,
        std_dev_factor: float = 0.25,
        support_radius: int = 2,
        **kwargs,
    ) -> None:
        """
        Smooth, compact-support bump around the middle layer with Gaussian-matched energy.

        Builds a Hann-window bump over indices within `support_radius` (default 2) from the
        center layer, zero outside. The bump is then scaled so that the total L1 energy equals
        the sum of coefficients from `set_control_with_gaussian_schedule` with the same
        `peak_coeff` and `std_dev_factor`.

        Intuition: approximates a Heaviside-like step over [middle-2, middle+2], but smooth.

        Args:
            control: The ControlVector to apply.
            peak_coeff: Peak coefficient used to compute reference Gaussian energy.
            std_dev_factor: Standard deviation factor for reference Gaussian energy.
            support_radius: Half-width of the compact support (in controlled index space).
            **kwargs: Additional arguments for set_raw_control.
        """
        import numpy as np

        n = len(self.layer_ids)
        if n == 0:
            return self.set_raw_control(None, **kwargs)

        # Reference total energy from Gaussian schedule (with center at floor(n/2) 1-based)
        if n == 1:
            total = float(peak_coeff)
        else:
            mu = n // 2
            std_dev = (n - 1) * float(std_dev_factor)
            if std_dev == 0:
                total = float(peak_coeff)
            else:
                pos = np.arange(1, n + 1, dtype=float)
                gaussian_vals = np.exp(-0.5 * ((pos - mu) / std_dev) ** 2)
                total = float(peak_coeff) * float(gaussian_vals.sum())

        # Build Hann bump around center within support_radius
        idx = np.arange(n, dtype=float)
        center = (n - 1) / 2.0
        d = np.abs(idx - center)
        R = max(float(support_radius), 1e-8)
        bump = np.where(d <= R, 0.5 * (1.0 + np.cos(np.pi * d / R)), 0.0)

        s = bump.sum()
        if s <= 0:
            # Degenerate case (e.g., extremely small n and zero radius) -> put all energy at center
            bump = np.zeros(n, dtype=float)
            center_i = int(round(center)) if n > 0 else 0
            bump[center_i] = 1.0
            s = 1.0

        coeffs = (total / s) * bump  # L1 sum equals 'total'

        def fn(i: int, _n: int) -> float:
            return float(coeffs[i])

        self.set_control(control, fn, **kwargs)

    def set_control_on_middle_layer(
        self,
        control: "ControlVector",
        max_coeff: float,
        **kwargs,
    ) -> None:
        """
        Set a `ControlVector` to apply only on the middle controlled layer.

        Args:
            control: The ControlVector to apply.
            max_coeff: The coefficient to apply at the middle layer, which also determines the sign.
            **kwargs: Additional arguments for set_raw_control.
        """

        def middle_fn(i, n):
            if n == 1:
                return max_coeff
            middle_index = (n - 1) // 2
            return max_coeff if i == middle_index else 0.0

        self.set_control(control, middle_fn, **kwargs)

    def set_control_with_sigmoid_schedule(
        self,
        control: "ControlVector",
        max_coeff: float,
        schedule_type: typing.Literal["early", "middle", "late"] = "middle",
        **kwargs,
    ) -> None:
        """
        Set a `ControlVector` with a sigmoid-based schedule.

        This is a "zero-hyperparameter" method in the sense that the shape of the
        control is determined automatically by the layer positions.

        Args:
            control: The ControlVector to apply.
            max_coeff: The maximum coefficient, which also determines the sign (direction).
            schedule_type: The type of schedule to use ('early', 'middle', 'late').
                           (default: 'middle')
            **kwargs: Additional arguments for set_raw_control.
        """
        import numpy as np

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_fn(i, n):
            if n == 1:
                return max_coeff

            # Create a scaled and shifted linear space for the sigmoid input
            x = np.linspace(-10, 10, n)

            if schedule_type == "early":
                # Strongest at the beginning, decays
                y = 1 - sigmoid(x)
            elif schedule_type == "late":
                # Starts weak, becomes strong
                y = sigmoid(x)
            elif schedule_type == "middle":
                # Peaks in the middle (approximates a bell curve)
                y = sigmoid(x) * (1 - sigmoid(x)) * 4
            else:
                raise ValueError(f"Unknown schedule_type: {schedule_type}")

            # Normalize the curve to have a max of 1 before scaling by max_coeff
            if np.max(y) > 0:
                y = y / np.max(y)

            return max_coeff * y[i]

        self.set_control(control, sigmoid_fn, **kwargs)

    def set_control_with_iterative_add_one_layer_schedule(
        self,
        control: "ControlVector",
        max_coeff: float,
        step: int,
        **kwargs,
    ) -> None:
        """
        Set a `ControlVector` with an iterative add-one-layer schedule.

        At a given step `k`, the `k` latest layers are active, and their
        coefficients are set to `max_coeff / (2^k)`.

        Args:
            control: The ControlVector to apply.
            max_coeff: The base coefficient, which also determines the sign.
            step: The iteration step `k`. Must be between 1 and the number of layers.
            **kwargs: Additional arguments for set_raw_control.
        """

        def iterative_fn(i, n):
            if step < 1 or step > n:
                raise ValueError(f"Step k must be between 1 and {n}, but got {step}")

            # The k latest layers are active.
            # Layer indices are 0 to n-1. The k latest layers start at index n-k.
            if i >= n - step:
                return max_coeff / (2**step)
            else:
                return 0.0

        self.set_control(control, iterative_fn, **kwargs)

    def set_control_direct(
        self,
        control: "ControlVector",
        **kwargs,
    ) -> None:
        """
        Apply a ControlVector *exactly as stored* (coeff = 1 everywhere).
        Good for eigen-scaled vectors where magnitude already encodes strength.
        """
        self.set_control_constant_scheduler(control, coeff=1.0, **kwargs)

    def set_control_constant_scheduler(
        self,
        control: "ControlVector",
        coeff: float,
        **kwargs,
    ) -> None:
        """
        Set a `ControlVector` with a constant coefficient across all layers.
        """

        def constant_fn(i, n):
            return coeff

        self.set_control(control, constant_fn, **kwargs)

    def set_control_energy_gaussian_schedule(
        self,
        control_vector: "ControlVector",
        tau: float = 1.0,
        normalize: str = "minmax",  # {"none","linf","l1","minmax"}
        sigma_floor: float = 1e-6,
        **kwargs,
    ) -> None:
        """
        Energy-weighted Gaussian schedule over *controlled order* with normalization.

        normalize:
        - "none":   no rescaling (raw Gaussian)
        - "linf":   divide by max so max==1
        - "l1":     divide by sum so coefficients sum to 1
        - "minmax": affine map to [0,1] across the discrete support

        Notes:
        - μ, σ are computed in the *controlled index* space (0..n-1),
            weighted by per-layer energy ||v_i||^2.
        - σ is floored for numerical stability.
        """
        import numpy as np

        # the layers we are actually controlling, in the same order set_control will iterate
        controlled_ids = list(self.layer_ids)
        if not all(l in control_vector.directions for l in controlled_ids):
            missing = [l for l in controlled_ids if l not in control_vector.directions]
            raise ValueError(f"Control vector missing directions for layers: {missing}")

        # index space is 0..n-1 in this exact order
        n = len(controlled_ids)
        idx = np.arange(n, dtype=float)
        e = np.array(
            [np.linalg.norm(control_vector.directions[l]) ** 2 for l in controlled_ids],
            dtype=float,
        )

        # energy-weighted center and width (in controlled index space)
        denom = e.sum() + 1e-12
        mu = float((idx * e).sum() / denom)
        var = float(((idx - mu) ** 2 * e).sum() / denom)
        sigma = max(np.sqrt(var), sigma_floor)

        raw = np.exp(-0.5 * ((idx - mu) / sigma) ** 2)

        # normalization to [0,1] (or sum==1)
        if normalize == "none":
            coeffs = raw
        elif normalize == "linf":
            coeffs = raw / (raw.max() + 1e-12)
        elif normalize == "l1":
            coeffs = raw / (raw.sum() + 1e-12)
        elif normalize == "minmax":
            rmin, rmax = raw.min(), raw.max()
            coeffs = (raw - rmin) / (rmax - rmin + 1e-12)
        else:
            raise ValueError(f"Unknown normalize='{normalize}'")

        # global scale
        coeffs = tau * coeffs

        def fn(i, n_):
            return float(coeffs[i])

        self.set_control(control_vector, fn, **kwargs)

    def set_control_energy_laplace_schedule(
        self,
        control_vector: "ControlVector",
        tau: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Energy-centered Laplace (double-exponential) schedule.
        Width is set from energy-weighted σ; coefficients are linf-normalized to [0,1],
        then scaled by tau. Single hyperparam: tau.
        """
        import numpy as np

        controlled_ids = list(self.layer_ids)
        if not all(l in control_vector.directions for l in controlled_ids):
            missing = [l for l in controlled_ids if l not in control_vector.directions]
            raise ValueError(f"Control vector missing directions for layers: {missing}")

        n = len(controlled_ids)
        idx = np.arange(n, dtype=float)
        e = np.array(
            [np.linalg.norm(control_vector.directions[l]) ** 2 for l in controlled_ids],
            dtype=float,
        )

        denom = e.sum() + 1e-12
        mu = float((idx * e).sum() / denom)
        var = float(((idx - mu) ** 2 * e).sum() / denom)
        sigma = max(np.sqrt(var), 1e-8)

        # Laplace variance = 2 b^2  =>  b = sigma / sqrt(2)
        b = max(sigma / np.sqrt(2.0), 1e-8)
        raw = np.exp(-np.abs(idx - mu) / b)
        coeffs = tau * (raw / (raw.max() + 1e-12))

        def fn(i, n_):
            return float(coeffs[i])

        self.set_control(control_vector, fn, **kwargs)

    def set_control_energy_hann_schedule(
        self,
        control_vector: "ControlVector",
        tau: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Energy-centered raised-cosine (Hann) window.
        Support radius comes from the Gaussian FWHM (≈2.355σ), giving smooth finite support.
        Coeffs are already in [0,1] at the center; final scale by tau. Single hyperparam: tau.
        """
        import numpy as np

        controlled_ids = list(self.layer_ids)
        if not all(l in control_vector.directions for l in controlled_ids):
            missing = [l for l in controlled_ids if l not in control_vector.directions]
            raise ValueError(f"Control vector missing directions for layers: {missing}")

        n = len(controlled_ids)
        idx = np.arange(n, dtype=float)
        e = np.array(
            [np.linalg.norm(control_vector.directions[l]) ** 2 for l in controlled_ids],
            dtype=float,
        )

        denom = e.sum() + 1e-12
        mu = float((idx * e).sum() / denom)
        var = float(((idx - mu) ** 2 * e).sum() / denom)
        sigma = max(np.sqrt(var), 1e-8)

        # Use half the Gaussian FWHM as radius: R = FWHM/2 = 1.17741 * sigma
        R = max(1.1774100225 * sigma, 1e-8)

        # Hann window on |i - mu| <= R ; zero outside
        d = np.abs(idx - mu)
        raw = np.where(d <= R, 0.5 * (1.0 + np.cos(np.pi * d / R)), 0.0)
        coeffs = tau * raw  # already [0,1]

        def fn(i, n_):
            return float(coeffs[i])

        self.set_control(control_vector, fn, **kwargs)

    def set_control_for_a_given_layer(
        self,
        control: "ControlVector",
        layer_id: int,
        coeff: float,
        **kwargs,
    ) -> None:
        """
        Set a `ControlVector` to apply only on a specific controlled layer.

        Args:
            control: The ControlVector to apply.
            layer_id: The specific layer ID to apply the control to.
            coeff: The coefficient to apply at the specified layer, which also determines the sign.
            **kwargs: Additional arguments for set_raw_control.
        """
        if layer_id not in self.layer_ids:
            raise ValueError(
                f"Layer ID {layer_id} is not in the controlled layer IDs: {self.layer_ids}"
            )

        def specific_layer_fn(i, n):
            if self.layer_ids[i] == layer_id:
                return coeff
            else:
                return 0.0

        self.set_control(control, specific_layer_fn, **kwargs)

    def set_control_directional_cosine_gated_for_layer(
        self,
        control: "ControlVector",
        layer_id: int,
        coeff: float,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        """
        Single-layer, signed cosine-gated steering:
        h' = h + cos_sim(h, v) * (coeff * v)
        """
        import torch
        import torch.nn.functional as F

        def cosine_gated_operator(
            current: torch.Tensor, add_vec: torch.Tensor
        ) -> torch.Tensor:
            # current, add_vec: [B, T, D]
            h_hat = F.normalize(current, p=2, dim=-1, eps=eps)
            v_hat = F.normalize(add_vec, p=2, dim=-1, eps=eps)
            cos = (h_hat * v_hat).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            return current + cos * add_vec

        return self.set_control_for_a_given_layer(
            control, layer_id, coeff, operator=cosine_gated_operator, **kwargs
        )

    def set_control_directional_relu_gated_for_layer(
        self,
        control: "ControlVector",
        layer_id: int,
        coeff: float,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        """
        Single-layer, one-sided (ReLU) cosine gate:
        h' = h + max(0, cos_sim(h, v)) * (coeff * v)
        Prevents harmful sign flips when anti-aligned.
        """
        import torch
        import torch.nn.functional as F

        def relu_gated_operator(
            current: torch.Tensor, add_vec: torch.Tensor
        ) -> torch.Tensor:
            h_hat = F.normalize(current, p=2, dim=-1, eps=eps)
            v_hat = F.normalize(add_vec, p=2, dim=-1, eps=eps)
            cos = (h_hat * v_hat).sum(dim=-1, keepdim=True)
            gate = torch.clamp(cos, min=0.0)
            return current + gate * add_vec

        return self.set_control_for_a_given_layer(
            control, layer_id, coeff, operator=relu_gated_operator, **kwargs
        )

    def set_control_directional_sigmoid_gated_for_layer(
        self,
        control: "ControlVector",
        layer_id: int,
        coeff: float,
        beta: float = 8.0,
        center: float = 0.0,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        """
        Single-layer, soft (sigmoid) cosine gate:
        g = sigmoid(beta * (cos_sim(h, v) - center))
        h' = h + g * (coeff * v)
        Smooth, thresholdable, never inverts direction.
        """
        import torch
        import torch.nn.functional as F

        def sigmoid_gated_operator(
            current: torch.Tensor, add_vec: torch.Tensor
        ) -> torch.Tensor:
            h_hat = F.normalize(current, p=2, dim=-1, eps=eps)
            v_hat = F.normalize(add_vec, p=2, dim=-1, eps=eps)
            cos = (h_hat * v_hat).sum(dim=-1, keepdim=True)
            gate = torch.sigmoid(beta * (cos - center))
            return current + gate * add_vec

        return self.set_control_for_a_given_layer(
            control, layer_id, coeff, operator=sigmoid_gated_operator, **kwargs
        )

    def set_constant_control_for_subset_of_layers(
        self,
        control: "ControlVector",
        layer_ids: typing.Iterable[int],
        coeff: float,
        **kwargs,
    ) -> None:
        """
        Set a `ControlVector` with a constant coefficient on a subset of controlled layers.

        Args:
            control: The ControlVector to apply.
            layer_ids: The specific layer IDs to apply the control to.
            coeff: The coefficient to apply at the specified layers, which also determines the sign.
            **kwargs: Additional arguments for set_raw_control.
        """
        for layer_id in layer_ids:
            if layer_id not in self.layer_ids:
                raise ValueError(
                    f"Layer ID {layer_id} is not in the controlled layer IDs: {self.layer_ids}"
                )

        def subset_fn(i, n):
            if self.layer_ids[i] in layer_ids:
                return coeff
            else:
                return 0.0

        self.set_control(control, subset_fn, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@dataclasses.dataclass
class BlockControlParams:
    control: torch.Tensor | None = None
    normalize: bool = False
    operator: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        return cls()


class ControlModule(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.block: torch.nn.Module = block
        self.params: BlockControlParams = BlockControlParams.default()

        if hasattr(block, "attention_type"):
            self.attention_type = block.attention_type

    def set_control(self, params: BlockControlParams) -> None:
        self.params = params

    def reset(self) -> None:
        self.set_control(BlockControlParams.default())

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        control = self.params.control

        if control is None:
            return output
        elif len(control.shape) == 1:
            control = control.reshape(1, 1, -1)

        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output

        assert len(control.shape) == len(modified.shape)
        control = control.to(modified.device)

        norm_pre = torch.norm(modified, dim=-1, keepdim=True)

        # we should ignore the padding tokens when doing the activation addition
        # mask has ones for non padding tokens and zeros at padding tokens.
        # only tested this on left padding
        if "position_ids" in kwargs:
            pos = kwargs["position_ids"]
            zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
            col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
            target_shape = modified.shape
            mask = (
                (col_indices >= zero_indices)
                .float()
                .reshape(target_shape[0], target_shape[1], 1)
            )
            mask = mask.to(modified.dtype).to(modified.device)
        else:
            mask = 1.0

        modified = self.params.operator(modified, control * mask)

        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output


def model_layer_list(model: ControlModel | PreTrainedModel) -> torch.nn.ModuleList:
    if isinstance(model, ControlModel):
        model = model.model

    if hasattr(model, "model"):  # mistral-like
        return model.model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")
