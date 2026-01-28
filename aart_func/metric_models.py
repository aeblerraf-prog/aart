from dataclasses import dataclass

from params import metric_model


@dataclass(frozen=True)
class MetricModel:
    name: str


def ensure_kerr_metric(context: str) -> None:
    if metric_model != "kerr":
        raise NotImplementedError(
            f"Metric model '{metric_model}' is not implemented for {context}. "
            "AART's analytic ray-tracing depends on Kerr integrability. "
            "To add Damour-Solodukhin or other wormhole metrics, implement the "
            "geodesic equations and conserved quantities in raytracing_f.py, then "
            "wire the new metric into metric_models.py."
        )


def get_metric_model() -> MetricModel:
    if metric_model != "kerr":
        raise NotImplementedError(
            f"Metric model '{metric_model}' is not implemented. "
            "AART currently supports Kerr only."
        )
    return MetricModel(name="kerr")
