"""Run every stage in order: checks -> v1 -> monitoring/v2 -> holdout -> findings.

Run:  python -m ev_energy.pipeline
"""

from .config import load_params
from .findings import write_findings
from .monitoring import run_monitoring
from .predict import predict_holdout
from .train import train_v1


def main() -> None:
    params = load_params()
    train_v1(params)
    run_monitoring(params)
    predict_holdout(params)
    write_findings(params)


if __name__ == "__main__":
    main()
