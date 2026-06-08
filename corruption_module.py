"""
corruption_module.py
Corrupts raw sensor data at the earliest possible stage (radar ADC, camera RGB).
Used to test system resilience: degraded health, missed detections, cascading failures.

Radar corruption modes (radar_corruption param):
  - 'gaussian':      additive complex Gaussian noise on ADC
  - 'blockage':      multiply all ADC arrays by attenuation factor [0, 1]
  - 'interference':  add complex linear chirp to N of 256 chirps
  - 'misalignment':  progressive phase shift across 16 Rx channels

Camera corruption modes (camera_corruption param):
  - 'gaussian': additive Gaussian noise on RGB
  - 'rain':     albumentations RandomRain
  - 'fog':      albumentations RandomFog
  - 'rain_fog': both rain + fog stacked

Usage:
    cm = CorruptionModule(radar_corruption='interference')
    cm.enable_radar = 1
    cm.radar_interference_chirps = 30
    cm.radar_interference_power = 3.0
    loader = RadialLoader(..., corruption_module=cm)
"""
import numpy as np


class CorruptionModule:
    def __init__(self, radar_enabled=0, camera_enabled=0,
                 radar_corruption='gaussian', camera_corruption='gaussian',
                 seed=42):
        """
        Parameters:
        -----------
        radar_enabled      : 1 = corrupt radar ADC, 0 = passthrough
        camera_enabled     : 1 = corrupt camera RGB, 0 = passthrough
        radar_corruption   : 'gaussian', 'blockage', 'interference', 'misalignment'
        camera_corruption  : 'gaussian', 'rain', 'fog', 'rain_fog'
        seed               : RNG seed for reproducible corruption
        """
        self.enable_radar = radar_enabled
        self.enable_camera = camera_enabled
        self.radar_corruption = radar_corruption
        self.camera_corruption = camera_corruption
        self.rng = np.random.default_rng(seed)

        # --- Gaussian noise params ---
        self.radar_noise_std = 2.0
        self.camera_noise_std = 40.0

        # --- Blockage params ---
        self.radar_blockage_factor = 0.3   # 0.0 = full block, 1.0 = no effect

        # --- Interference params ---
        self.radar_interference_chirps = 30   # how many of 256 chirps get interference
        self.radar_interference_power = 3.0   # amplitude relative to signal RMS

        # --- Misalignment params ---
        self.radar_misalignment_deg = 10.0   # apparent misalignment angle

        # ADC constants (from RADIal rpl.py: numSamplePerChirp=512, numChirps=256, numRxPerChip=4)
        self._NUM_SAMPLES = 512
        self._NUM_CHIRPS = 256
        self._NUM_RX_PER_CHIP = 4

        # Albumentations transforms (lazy-loaded)
        self._rain_transform = None
        self._fog_transform = None

    # -----------------------------------------------------------------------
    # Radar corruption — applied to raw ADC samples (4 chips)
    # -----------------------------------------------------------------------
    def corrupt_radar_adc(self, adc0, adc1, adc2, adc3):
        if not self.enable_radar:
            return adc0, adc1, adc2, adc3

        if self.radar_corruption == 'gaussian':
            return self._apply_gaussian_radar(adc0, adc1, adc2, adc3)
        elif self.radar_corruption == 'blockage':
            return self._apply_blockage(adc0, adc1, adc2, adc3)
        elif self.radar_corruption == 'interference':
            return self._apply_interference(adc0, adc1, adc2, adc3)
        elif self.radar_corruption == 'misalignment':
            return self._apply_misalignment(adc0, adc1, adc2, adc3)
        else:
            return adc0, adc1, adc2, adc3

    # -----------------------------------------------------------------------
    # Helper: ADC ↔ complex frame conversion
    # -----------------------------------------------------------------------
    def _adc_to_complex(self, adc):
        """Convert raw int16 ADC to complex frame (samples, chirps, rx)."""
        complex_1d = adc[0::2].astype(np.float32) + 1j * adc[1::2].astype(np.float32)
        frame = np.reshape(complex_1d,
            (self._NUM_SAMPLES, self._NUM_RX_PER_CHIP, self._NUM_CHIRPS),
            order='F')
        return frame.transpose((0, 2, 1))  # (samples, chirps, rx)

    def _complex_to_adc(self, frame):
        """Convert complex frame (samples, chirps, rx) back to raw int16 ADC."""
        frame = np.ascontiguousarray(frame.transpose((0, 2, 1)))  # (samples, rx, chirps)
        flat = frame.ravel(order='F')  # (samples*rx*chirps,) complex
        out = np.empty(flat.shape[0] * 2, dtype=np.int16)
        out[0::2] = np.clip(np.round(flat.real), -32768, 32767).astype(np.int16)
        out[1::2] = np.clip(np.round(flat.imag), -32768, 32767).astype(np.int16)
        return out

    # -----------------------------------------------------------------------
    # 1) Blockage — attenuate all ADC by factor
    # -----------------------------------------------------------------------
    def _apply_blockage(self, adc0, adc1, adc2, adc3):
        f = self.radar_blockage_factor
        def _scale(adc):
            return (adc.astype(np.float32) * f).astype(np.int16)
        return _scale(adc0), _scale(adc1), _scale(adc2), _scale(adc3)

    # -----------------------------------------------------------------------
    # 2) Interference — add complex linear chirp to selected chirps
    # -----------------------------------------------------------------------
    def _apply_interference(self, adc0, adc1, adc2, adc3):
        # Convert each chip to complex frame
        frames = [self._adc_to_complex(adc) for adc in (adc0, adc1, adc2, adc3)]

        # Build chirp signal: exp(j * pi * rate * (n/N)^2)
        n = np.arange(self._NUM_SAMPLES, dtype=np.float32) / self._NUM_SAMPLES
        chirp_signal = np.exp(1j * np.pi * 8.0 * n ** 2)  # 8x bandwidth factor

        # Select random chirp indices to corrupt
        num_affected = min(self.radar_interference_chirps, self._NUM_CHIRPS)
        affected_chirps = self.rng.choice(
            self._NUM_CHIRPS, size=num_affected, replace=False)

        # Compute interference amplitude relative to each chip's RMS
        for i in range(4):
            rms = np.sqrt(np.mean(np.abs(frames[i]) ** 2))
            amplitude = rms * self.radar_interference_power
            # Add chirp to all Rx channels for selected chirps
            frames[i][:, affected_chirps, :] += amplitude * chirp_signal[:, np.newaxis, np.newaxis]

        # Convert back to ADC
        return tuple(self._complex_to_adc(f) for f in frames)

    # -----------------------------------------------------------------------
    # 3) Misalignment — progressive phase shift across 16 Rx channels
    #    For a uniform linear array with d = λ/2 (standard 77 GHz automotive radar):
    #      Δφ = π * sin(θ)  where θ is the misalignment angle
    # -----------------------------------------------------------------------
    def _apply_misalignment(self, adc0, adc1, adc2, adc3):
        # Global Rx ordering from rpl.py: [chip3, chip0, chip1, chip2]
        chips = [adc3, adc0, adc1, adc2]
        frames = [self._adc_to_complex(adc) for adc in chips]

        theta_rad = np.radians(self.radar_misalignment_deg)
        # Phase shift per Rx element for d = λ/2
        phase_per_rx = np.pi * np.sin(theta_rad)

        for global_rx in range(16):
            chip_idx = global_rx // 4
            rx_in_chip = global_rx % 4
            shift = np.exp(1j * global_rx * phase_per_rx)
            frames[chip_idx][:, :, rx_in_chip] *= shift

        # Convert back: frames are [chip3, chip0, chip1, chip2]
        adc3_out = self._complex_to_adc(frames[0])
        adc0_out = self._complex_to_adc(frames[1])
        adc1_out = self._complex_to_adc(frames[2])
        adc2_out = self._complex_to_adc(frames[3])
        return adc0_out, adc1_out, adc2_out, adc3_out

    # -----------------------------------------------------------------------
    # 4) Gaussian noise (original)
    # -----------------------------------------------------------------------
    def _apply_gaussian_radar(self, adc0, adc1, adc2, adc3):
        out = []
        for adc in (adc0, adc1, adc2, adc3):
            noise_std = self.radar_noise_std * np.std(adc.astype(np.float32))
            noise = self.rng.normal(0, noise_std, adc.shape).astype(np.float32)
            out.append((adc.astype(np.float32) + noise).astype(np.int16))
        return tuple(out)

    # -----------------------------------------------------------------------
    # Camera corruption — applied to raw RGB image
    # -----------------------------------------------------------------------
    def corrupt_camera_image(self, image):
        if not self.enable_camera:
            return image

        if self.camera_corruption == 'gaussian':
            return self._apply_gaussian_camera(image)
        elif self.camera_corruption == 'rain':
            return self._apply_rain(image)
        elif self.camera_corruption == 'fog':
            return self._apply_fog(image)
        elif self.camera_corruption == 'rain_fog':
            image = self._apply_rain(image)
            image = self._apply_fog(image)
            return image
        else:
            return image

    def _apply_gaussian_camera(self, image):
        noise = self.rng.normal(0, self.camera_noise_std, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def _get_rain_transform(self):
        if self._rain_transform is None:
            import albumentations as A
            self._rain_transform = A.Compose([
                A.RandomRain(
                    slant_range=(-10, 10),
                    drop_length=20, drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=3, brightness_coefficient=0.8,
                    rain_type="drizzle",
                    p=1.0
                )
            ])
        return self._rain_transform

    def _get_fog_transform(self):
        if self._fog_transform is None:
            import albumentations as A
            self._fog_transform = A.Compose([
                A.RandomFog(
                    fog_coef_range=(0.3, 0.7),
                    alpha_coef=0.08,
                    p=1.0
                )
            ])
        return self._fog_transform

    def _apply_rain(self, image):
        return self._get_rain_transform()(image=image)["image"]

    def _apply_fog(self, image):
        return self._get_fog_transform()(image=image)["image"]

    # -----------------------------------------------------------------------
    # Convenience toggles
    # -----------------------------------------------------------------------
    def enable_radar_only(self):
        self.enable_radar = 1
        self.enable_camera = 0

    def enable_camera_only(self):
        self.enable_radar = 0
        self.enable_camera = 1

    def enable_both(self):
        self.enable_radar = 1
        self.enable_camera = 1

    def disable_all(self):
        self.enable_radar = 0
        self.enable_camera = 0
