"""Unit tests for ComfyUI scheduler factory (comfyui_fluxflow/schedulers.py)."""

import pytest

from comfyui_fluxflow.schedulers import (
    ALGORITHM_TYPES,
    PREDICTION_TYPES,
    SCHEDULER_DEFAULTS,
    SCHEDULER_NAMES,
    _check_diffusers_available,
    create_scheduler,
    get_scheduler_info,
    get_scheduler_list,
)


class TestSchedulerConfiguration:
    """Tests for scheduler configuration constants."""

    def test_scheduler_names_list(self):
        """SCHEDULER_NAMES should contain 14 schedulers."""
        assert len(SCHEDULER_NAMES) == 14
        assert "DPMSolverMultistep" in SCHEDULER_NAMES
        assert "EulerDiscrete" in SCHEDULER_NAMES
        assert "DDIM" in SCHEDULER_NAMES

    def test_all_schedulers_have_defaults(self):
        """All schedulers should have default configurations."""
        for scheduler_name in SCHEDULER_NAMES:
            assert scheduler_name in SCHEDULER_DEFAULTS

    def test_default_configs_have_prediction_type(self):
        """All default configs should specify prediction_type."""
        for scheduler_name, config in SCHEDULER_DEFAULTS.items():
            assert "prediction_type" in config
            assert config["prediction_type"] in PREDICTION_TYPES

    def test_prediction_types_list(self):
        """PREDICTION_TYPES should contain valid types."""
        assert len(PREDICTION_TYPES) == 3
        assert "v_prediction" in PREDICTION_TYPES
        assert "epsilon" in PREDICTION_TYPES
        assert "sample" in PREDICTION_TYPES

    def test_algorithm_types_list(self):
        """ALGORITHM_TYPES should contain DPM solver algorithms."""
        assert "dpmsolver++" in ALGORITHM_TYPES
        assert "dpmsolver" in ALGORITHM_TYPES
        assert "sde-dpmsolver++" in ALGORITHM_TYPES

    def test_dpm_solver_multistep_defaults(self):
        """DPMSolverMultistep should have comprehensive defaults."""
        config = SCHEDULER_DEFAULTS["DPMSolverMultistep"]

        assert config["algorithm_type"] == "dpmsolver++"
        assert config["solver_order"] == 2
        assert config["prediction_type"] == "v_prediction"
        assert config["lower_order_final"] is True
        assert config["timestep_spacing"] == "trailing"

    def test_euler_discrete_defaults(self):
        """EulerDiscrete should have correct defaults."""
        config = SCHEDULER_DEFAULTS["EulerDiscrete"]

        assert config["prediction_type"] == "v_prediction"
        assert config["timestep_spacing"] == "trailing"

    def test_ddim_defaults(self):
        """DDIM should have correct defaults."""
        config = SCHEDULER_DEFAULTS["DDIM"]

        assert config["prediction_type"] == "v_prediction"
        assert config["clip_sample"] is False


class TestGetSchedulerList:
    """Tests for get_scheduler_list function."""

    def test_returns_list(self):
        """Should return a list of scheduler names."""
        schedulers = get_scheduler_list()

        assert isinstance(schedulers, list)
        assert len(schedulers) > 0

    def test_returns_all_schedulers(self):
        """Should return all 14 schedulers."""
        schedulers = get_scheduler_list()

        assert len(schedulers) == 14

    def test_contains_common_schedulers(self):
        """Should contain commonly used schedulers."""
        schedulers = get_scheduler_list()

        assert "DPMSolverMultistep" in schedulers
        assert "EulerDiscrete" in schedulers
        assert "DDIM" in schedulers
        assert "LCM" in schedulers

    def test_contains_karras_scheduler(self):
        """Should include DPMPlusPlusKarras scheduler."""
        schedulers = get_scheduler_list()

        assert "DPMPlusPlusKarras" in schedulers


class TestCreateScheduler:
    """Tests for create_scheduler function."""

    def test_create_dpm_solver_multistep(self):
        """Should create DPMSolverMultistep scheduler."""
        scheduler = create_scheduler("DPMSolverMultistep")

        assert scheduler is not None
        # Check it has necessary methods
        assert hasattr(scheduler, "set_timesteps")

    def test_create_euler_discrete(self):
        """Should create EulerDiscrete scheduler."""
        scheduler = create_scheduler("EulerDiscrete")

        assert scheduler is not None
        assert hasattr(scheduler, "set_timesteps")

    def test_create_ddim(self):
        """Should create DDIM scheduler."""
        scheduler = create_scheduler("DDIM")

        assert scheduler is not None
        assert hasattr(scheduler, "set_timesteps")

    def test_create_all_schedulers(self):
        """Should be able to create all 14 schedulers."""
        for scheduler_name in SCHEDULER_NAMES:
            scheduler = create_scheduler(scheduler_name)
            assert scheduler is not None
            assert hasattr(scheduler, "set_timesteps")

    def test_custom_num_train_timesteps(self):
        """Should accept custom num_train_timesteps."""
        scheduler = create_scheduler("DPMSolverMultistep", num_train_timesteps=500)

        # Check timesteps were set correctly
        assert hasattr(scheduler, "config")
        if hasattr(scheduler.config, "num_train_timesteps"):
            assert scheduler.config.num_train_timesteps == 500

    def test_custom_prediction_type(self):
        """Should accept custom prediction_type."""
        scheduler = create_scheduler("DPMSolverMultistep", prediction_type="epsilon")

        assert hasattr(scheduler, "config")
        if hasattr(scheduler.config, "prediction_type"):
            assert scheduler.config.prediction_type == "epsilon"

    def test_additional_kwargs(self):
        """Should accept additional kwargs."""
        scheduler = create_scheduler("DPMSolverMultistep", num_train_timesteps=1000, solver_order=3)

        assert scheduler is not None

    def test_invalid_scheduler_raises_error(self):
        """Should raise ValueError for unknown scheduler."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler("InvalidSchedulerName")

    def test_default_num_train_timesteps(self):
        """Should use 1000 as default num_train_timesteps."""
        scheduler = create_scheduler("DPMSolverMultistep")

        assert hasattr(scheduler, "config")
        if hasattr(scheduler.config, "num_train_timesteps"):
            assert scheduler.config.num_train_timesteps == 1000


class TestGetSchedulerInfo:
    """Tests for get_scheduler_info function."""

    def test_get_info_for_dpm_solver(self):
        """Should return info for DPMSolverMultistep."""
        info = get_scheduler_info("DPMSolverMultistep")

        assert "class" in info
        assert "defaults" in info
        assert info["defaults"]["algorithm_type"] == "dpmsolver++"

    def test_get_info_for_euler(self):
        """Should return info for EulerDiscrete."""
        info = get_scheduler_info("EulerDiscrete")

        assert "class" in info
        assert "defaults" in info
        assert info["defaults"]["prediction_type"] == "v_prediction"

    def test_get_info_for_all_schedulers(self):
        """Should return info for all schedulers."""
        for scheduler_name in SCHEDULER_NAMES:
            info = get_scheduler_info(scheduler_name)
            assert "class" in info
            assert "defaults" in info

    def test_invalid_scheduler_raises_error(self):
        """Should raise ValueError for unknown scheduler."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler_info("InvalidScheduler")

    def test_info_class_is_callable(self):
        """Scheduler class should be callable."""
        info = get_scheduler_info("DPMSolverMultistep")

        assert callable(info["class"])

    def test_info_defaults_match_config(self):
        """Info defaults should match SCHEDULER_DEFAULTS."""
        for scheduler_name in SCHEDULER_NAMES:
            info = get_scheduler_info(scheduler_name)
            assert info["defaults"] == SCHEDULER_DEFAULTS[scheduler_name]


class TestDiffusersAvailability:
    """Tests for diffusers availability check."""

    def test_check_diffusers_available(self):
        """Should check if diffusers is available."""
        result = _check_diffusers_available()

        # Result should be boolean
        assert isinstance(result, bool)

    def test_check_diffusers_consistent(self):
        """Should return consistent result on multiple calls."""
        result1 = _check_diffusers_available()
        result2 = _check_diffusers_available()

        assert result1 == result2


class TestSchedulerIntegration:
    """Integration tests for scheduler usage."""

    def test_create_and_set_timesteps(self):
        """Should create scheduler and set timesteps."""
        scheduler = create_scheduler("DPMSolverMultistep")

        # Set timesteps (this would normally use device)

        scheduler.set_timesteps(20, device="cpu")

        # Should have timesteps attribute
        assert hasattr(scheduler, "timesteps")
        assert len(scheduler.timesteps) > 0

    def test_multiple_schedulers_independent(self):
        """Multiple scheduler instances should be independent."""
        scheduler1 = create_scheduler("DPMSolverMultistep", num_train_timesteps=500)
        scheduler2 = create_scheduler("DPMSolverMultistep", num_train_timesteps=1000)

        # Should be different instances
        assert scheduler1 is not scheduler2

    def test_scheduler_for_sampling_workflow(self):
        """Should work in typical sampling workflow."""

        scheduler = create_scheduler("EulerDiscrete")
        scheduler.set_timesteps(10, device="cpu")

        # Should have timesteps
        assert len(scheduler.timesteps) == 10

        # Timesteps should be in valid range
        assert all(t >= 0 for t in scheduler.timesteps)

    def test_all_schedulers_support_set_timesteps(self):
        """All schedulers should support set_timesteps."""

        for scheduler_name in SCHEDULER_NAMES:
            scheduler = create_scheduler(scheduler_name)
            scheduler.set_timesteps(5, device="cpu")
            assert len(scheduler.timesteps) > 0

    def test_scheduler_with_v_prediction(self):
        """Should create scheduler with v_prediction."""
        scheduler = create_scheduler("DPMSolverMultistep", prediction_type="v_prediction")

        assert hasattr(scheduler, "config")

    def test_scheduler_with_epsilon_prediction(self):
        """Should create scheduler with epsilon prediction."""
        scheduler = create_scheduler("DDIM", prediction_type="epsilon")

        assert hasattr(scheduler, "config")

    def test_lcm_scheduler_creation(self):
        """Should create LCM scheduler for fast sampling."""
        scheduler = create_scheduler("LCM")

        assert scheduler is not None
        assert hasattr(scheduler, "set_timesteps")

    def test_karras_scheduler_creation(self):
        """Should create DPMPlusPlusKarras scheduler."""
        scheduler = create_scheduler("DPMPlusPlusKarras")

        assert scheduler is not None
        assert hasattr(scheduler, "set_timesteps")

    def test_ancestral_schedulers(self):
        """Should create ancestral (stochastic) schedulers."""
        ancestral_schedulers = [
            "EulerAncestralDiscrete",
            "KDPM2AncestralDiscrete",
        ]

        for scheduler_name in ancestral_schedulers:
            scheduler = create_scheduler(scheduler_name)
            assert scheduler is not None

    def test_multistep_schedulers(self):
        """Should create multistep schedulers."""
        multistep_schedulers = [
            "DPMSolverMultistep",
            "UniPCMultistep",
            "DEISMultistep",
        ]

        for scheduler_name in multistep_schedulers:
            scheduler = create_scheduler(scheduler_name)
            assert scheduler is not None


class TestSchedulerFallback:
    """Tests for fallback to standalone schedulers."""

    def test_fallback_schedulers_exist(self):
        """Standalone schedulers should exist as fallback."""
        from comfyui_fluxflow.standalone_schedulers import STANDALONE_SCHEDULERS

        assert "DDIM" in STANDALONE_SCHEDULERS
        assert "Euler" in STANDALONE_SCHEDULERS
        assert "DPMPlusPlus" in STANDALONE_SCHEDULERS
        assert "FlowMatching" in STANDALONE_SCHEDULERS

    def test_standalone_schedulers_have_required_params(self):
        """Standalone schedulers should accept num_train_timesteps."""
        from comfyui_fluxflow.standalone_schedulers import STANDALONE_SCHEDULERS

        for name, scheduler_cls in STANDALONE_SCHEDULERS.items():
            # Should be able to instantiate with basic params
            scheduler = scheduler_cls(num_train_timesteps=1000, prediction_type="v_prediction")
            assert scheduler is not None


class TestSchedulerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_scheduler_name(self):
        """Should handle empty scheduler name."""
        with pytest.raises(ValueError):
            create_scheduler("")

    def test_none_scheduler_name(self):
        """Should handle None scheduler name."""
        with pytest.raises((ValueError, TypeError)):
            create_scheduler(None)

    def test_case_sensitive_scheduler_name(self):
        """Scheduler names should be case-sensitive."""
        # Correct case should work
        scheduler = create_scheduler("DPMSolverMultistep")
        assert scheduler is not None

        # Wrong case should fail
        with pytest.raises(ValueError):
            create_scheduler("dpmSolverMultistep")

    def test_very_low_timesteps(self):
        """Should handle very low num_train_timesteps."""
        scheduler = create_scheduler("DDIM", num_train_timesteps=10)
        assert scheduler is not None

    def test_very_high_timesteps(self):
        """Should handle very high num_train_timesteps."""
        scheduler = create_scheduler("DDIM", num_train_timesteps=5000)
        assert scheduler is not None
