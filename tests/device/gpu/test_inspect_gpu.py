"""Tests for normalized GPU hardware inspection."""

from __future__ import annotations

from types import SimpleNamespace

import zeus.device.gpu as gpu_module
import zeus.device.gpu.amd as amd_module
import zeus.device.gpu.nvidia as nvidia_module
from zeus.device.gpu.common import ZeusGPUNotSupportedError


def test_inspect_gpu_dispatches_to_nvidia(mocker):
    """Dispatch inspection to the NVIDIA collector."""
    gpu = mocker.MagicMock()
    gpus = mocker.MagicMock(spec=nvidia_module.NVIDIAGPUs)
    gpus.__len__.return_value = 1
    gpus.gpus = [gpu]
    expected = object()
    mocker.patch("zeus.device.gpu.get_gpus", return_value=gpus)
    collector = mocker.patch("zeus.device.gpu.nvidia.inspect_gpu", return_value=expected)

    assert gpu_module.inspect_gpu() is expected
    collector.assert_called_once_with(gpu)


def test_nvidia_inspect_gpu_normalizes_hardware_info(mocker):
    """Normalize NVIDIA API values and units."""
    gpu = SimpleNamespace(
        gpu_index=3,
        handle="handle",
        get_name=mocker.MagicMock(return_value="NVIDIA Test GPU"),
        get_persistence_mode=mocker.MagicMock(return_value=True),
        get_gpu_temperature=mocker.MagicMock(return_value=42),
        get_instant_power_usage=mocker.MagicMock(return_value=120_500),
        get_power_management_limit=mocker.MagicMock(return_value=250_000),
        get_power_management_limit_constraints=mocker.MagicMock(return_value=(200_000, 300_000)),
        get_supported_memory_clocks=mocker.MagicMock(return_value=[2000, 1000]),
        get_supported_graphics_clocks=mocker.MagicMock(
            side_effect=lambda memory: [1800, 900] if memory == 2000 else [1500]
        ),
    )

    nvml = nvidia_module.pynvml
    mocker.patch.object(nvml, "nvmlSystemGetDriverVersion", return_value="555.1")
    mocker.patch.object(nvml, "nvmlSystemGetNVMLVersion", return_value="12.0")
    mocker.patch.object(nvml, "nvmlDeviceGetPciInfo", return_value=SimpleNamespace(busId="0000:01:00.0"))
    mocker.patch.object(nvml, "nvmlDeviceGetArchitecture", return_value=nvml.NVML_DEVICE_ARCH_AMPERE)
    mocker.patch.object(nvml, "nvmlDeviceGetCudaComputeCapability", return_value=(8, 0))
    mocker.patch.object(nvml, "nvmlDeviceGetMemoryInfo", return_value=SimpleNamespace(total=80 * 1024**3))
    mocker.patch.object(nvml, "nvmlDeviceGetPowerManagementDefaultLimit", return_value=275_000)
    mocker.patch.object(
        nvml,
        "nvmlDeviceGetClockInfo",
        side_effect=lambda _handle, clock: 1600 if clock == nvml.NVML_CLOCK_MEM else 1200,
    )

    info = nvidia_module.inspect_gpu(gpu)

    assert info.vendor == "nvidia"
    assert info.gpu_index == 3
    assert info.model_name == "NVIDIA Test GPU"
    assert info.pci_address == "0000:01:00.0"
    assert info.architecture == "ampere"
    assert info.compute_capability == (8, 0)
    assert info.total_memory_mb == 80 * 1024
    assert info.persistence_mode is True
    assert info.current_temperature_c == 42
    assert info.current_power_w == 120.5
    assert info.current_power_limit_w == 250.0
    assert info.default_power_limit_w == 275.0
    assert info.power_limit_range_w == (200.0, 300.0)
    assert info.current_memory_clock_mhz == 1600
    assert info.supported_memory_clocks_mhz == (1000, 2000)
    assert info.current_graphics_clock_mhz == 1200
    assert info.supported_graphics_clocks_mhz == (900, 1500, 1800)
    assert info.supported_graphics_clocks_by_memory_clock_mhz == {
        1000: (1500,),
        2000: (900, 1800),
    }


def test_amd_inspect_gpu_normalizes_hardware_info(mocker):
    """Normalize AMD API values and unsupported properties."""
    memory_clock = object()
    graphics_clock = object()
    amdsmi = SimpleNamespace(
        AmdSmiClkType=SimpleNamespace(MEM=memory_clock, GFX=graphics_clock),
        amdsmi_get_lib_version=mocker.MagicMock(return_value={"major": 6, "minor": 4, "release": 1, "build": "test"}),
        amdsmi_get_gpu_driver_info=mocker.MagicMock(return_value={"driver_version": "6.4"}),
        amdsmi_get_gpu_asic_info=mocker.MagicMock(return_value={"target_graphics_version": "gfx942"}),
        amdsmi_get_gpu_device_bdf=mocker.MagicMock(return_value="0000:41:00.0"),
        amdsmi_get_gpu_vram_info=mocker.MagicMock(return_value={"vram_size": 65_536}),
        amdsmi_get_power_cap_info=mocker.MagicMock(return_value={"default_power_cap": 325_000_000}),
        amdsmi_get_clock_info=mocker.MagicMock(
            side_effect=lambda _handle, clock: {"clk": 1400 if clock is memory_clock else 1900}
        ),
        amdsmi_get_clk_freq=mocker.MagicMock(
            side_effect=lambda _handle, clock: {
                "frequency": [800_000_000, 1_600_000_000]
                if clock is memory_clock
                else [500_000_000, 1_500_000_000, 2_100_000_000]
            }
        ),
    )
    mocker.patch.object(amd_module, "amdsmi", amdsmi)
    gpu = SimpleNamespace(
        gpu_index=2,
        handle="handle",
        get_name=mocker.MagicMock(return_value="AMD Test GPU"),
        get_persistence_mode=mocker.MagicMock(side_effect=ZeusGPUNotSupportedError("unsupported")),
        get_gpu_temperature=mocker.MagicMock(return_value=45),
        get_instant_power_usage=mocker.MagicMock(return_value=90_250),
        get_power_management_limit=mocker.MagicMock(return_value=300_000),
        get_power_management_limit_constraints=mocker.MagicMock(return_value=(200_000, 350_000)),
    )

    info = amd_module.inspect_gpu(gpu)

    assert info.vendor == "amd"
    assert info.gpu_index == 2
    assert info.model_name == "AMD Test GPU"
    assert info.software_versions == {"amdsmi": "6.4.1+test", "driver_version": "6.4"}
    assert info.pci_address == "0000:41:00.0"
    assert info.architecture == "gfx942"
    assert info.compute_capability is None
    assert info.total_memory_mb == 65_536
    assert info.persistence_mode is None
    assert info.current_temperature_c == 45
    assert info.current_power_w == 90.25
    assert info.current_power_limit_w == 300.0
    assert info.default_power_limit_w == 325.0
    assert info.power_limit_range_w == (200.0, 350.0)
    assert info.current_memory_clock_mhz == 1400
    assert info.supported_memory_clocks_mhz == (800, 1600)
    assert info.current_graphics_clock_mhz == 1900
    assert info.supported_graphics_clocks_mhz == (500, 1500, 2100)
    assert info.supported_graphics_clocks_by_memory_clock_mhz is None
