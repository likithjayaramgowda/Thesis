from dataclasses import dataclass
import psutil
import time


@dataclass
class EnergyProxyResult:
    wh_per_1000: float
    method: str


def measure_energy_proxy(duration_s: float, avg_cpu_util: float, tdp_watts: float) -> EnergyProxyResult:
    # Simple proxy: energy = util * TDP * time
    watts = (avg_cpu_util / 100.0) * tdp_watts
    wh = (watts * duration_s) / 3600.0
    return EnergyProxyResult(wh_per_1000=wh * (1000.0 / max(1, int(duration_s > 0))), method="cpu_util_proxy")


def sample_cpu_util(interval_s: float = 0.1, samples: int = 10) -> float:
    vals = []
    for _ in range(samples):
        vals.append(psutil.cpu_percent(interval=interval_s))
    return sum(vals) / len(vals)
