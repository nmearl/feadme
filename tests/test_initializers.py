from pathlib import Path

from feadme.core.parser import Template
from feadme.sampling.initializers import SVIInitializer, _structured_lsq_candidates


class _Config:
    def __init__(self, template):
        self.template = template


def test_structured_lsq_candidates_expand_to_requested_count():
    template = Template.from_json(
        Path(__file__).parents[1] / "scripts/synthetic_recovery_run/template.json"
    )

    candidates = _structured_lsq_candidates(_Config(template), target_count=48)

    assert len(candidates) == 48
    assert (
        len(
            {
                tuple(
                    sorted(
                        (key, round(float(value), 10)) for key, value in item.items()
                    )
                )
                for item in candidates
            }
        )
        == 48
    )


def test_svi_candidate_ranking_rejects_unstable_high_density_candidate():
    initializer = SVIInitializer(max_candidate_loss_relative_std=0.10)

    ranked = initializer._rank_candidates(
        [
            {
                "candidate_id": 0,
                "score": 720.0,
                "loss_relative_std": 0.15,
                "lsq_objective": 90.0,
            },
            {
                "candidate_id": 1,
                "score": 718.0,
                "loss_relative_std": 0.06,
                "lsq_objective": 96.0,
            },
        ]
    )

    assert ranked[0]["candidate_id"] == 1
    assert ranked[0]["selection_eligible"] is True
    assert ranked[0]["selection_rank"] == 0
    assert ranked[1]["selection_eligible"] is False
    assert ranked[1]["selection_rejection_reason"] == "loss_relative_std"


def test_svi_candidate_ranking_falls_back_when_no_candidate_is_stable():
    initializer = SVIInitializer(max_candidate_loss_relative_std=0.10)

    ranked = initializer._rank_candidates(
        [
            {
                "candidate_id": 0,
                "score": 720.0,
                "loss_relative_std": 0.15,
                "lsq_objective": 90.0,
            },
            {
                "candidate_id": 1,
                "score": 718.0,
                "loss_relative_std": 0.12,
                "lsq_objective": 96.0,
            },
        ]
    )

    assert ranked[0]["candidate_id"] == 0
    assert ranked[0]["selection_rejection_reason"] == "fallback_no_eligible_candidates"
