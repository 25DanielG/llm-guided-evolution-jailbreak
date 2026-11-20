"""Run the prepared deployment's benign probe with the guide validation client."""
from ops import pipeline_probe

from guide_fixtures import guide_label_client


if __name__ == "__main__":
    with guide_label_client():
        pipeline_probe.main()
