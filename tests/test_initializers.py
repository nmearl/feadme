import numpy as np

from feadme.core.parser import Disk, Distribution, Line, Parameter, Template
from feadme.sampling.initializers import SVIInitializer, _structured_lsq_candidates


class _Config:
    def __init__(self, template):
        self.template = template


def _param(distribution, low, high, loc, scale, *, fixed=False, circular=False):
    return Parameter(
        distribution=distribution,
        fixed=fixed,
        low=low,
        high=high,
        loc=loc,
        scale=scale,
        circular=circular,
    )


def _test_template():
    return Template.create(
        name="test_template",
        disk_profiles=[
            Disk(
                name="halpha_disk",
                center=6564.61,
                offset=_param(Distribution.NORMAL, -1.0e4, 1.0e4, 0.0, 600.0, fixed=True),
                inner_radius=_param(Distribution.LOG_UNIFORM, 100.0, 1500.0, 300.0, 50.0),
                outer_radius=_param(Distribution.LOG_UNIFORM, 1000.0, 2.0e4, 5000.0, 1000.0),
                inclination=_param(Distribution.UNIFORM, 0.0, np.pi / 2, 0.5, 0.2),
                sigma=_param(Distribution.LOG_UNIFORM, 100.0, 3000.0, 1000.0, 300.0),
                q=_param(Distribution.UNIFORM, 1.0, 4.0, 2.0, 0.5),
                eccentricity=_param(Distribution.UNIFORM, 0.0, 0.95, 0.2, 0.1),
                apocenter=_param(
                    Distribution.UNIFORM,
                    0.0,
                    2 * np.pi,
                    np.pi,
                    0.5,
                    circular=True,
                ),
                area=_param(Distribution.LOG_UNIFORM, 1.0, 100.0, 20.0, 5.0),
                baseline=_param(Distribution.UNIFORM, 0.0, 2.0, 0.0, 0.001, fixed=True),
            )
        ],
        line_profiles=[
            Line(
                name="halpha_broad",
                center=6564.61,
                offset=_param(Distribution.NORMAL, -2.0e3, 2.0e3, 0.0, 300.0),
                area=_param(Distribution.LOG_UNIFORM, 1.0, 50.0, 10.0, 2.0),
                vel_width=_param(Distribution.LOG_UNIFORM, 500.0, 5000.0, 1500.0, 300.0),
            )
        ],
        redshift=_param(Distribution.UNIFORM, 0.0, 0.2, 0.05, 0.01),
        log_frac_noise=_param(Distribution.UNIFORM, -10.0, 1.0, -6.0, 1.0),
    )


def test_structured_lsq_candidates_expand_to_requested_count():
    template = _test_template()

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
