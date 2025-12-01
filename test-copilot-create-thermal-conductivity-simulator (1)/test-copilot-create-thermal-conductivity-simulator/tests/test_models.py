"""
Unit tests for Nanofluid Thermal Conductivity Simulator
"""

import pytest
import math
from nanofluid_simulator.models import (
    maxwell_model,
    hamilton_crosser_model,
    bruggeman_model,
    yu_choi_model,
    wasp_model,
    pak_cho_correlation,
    validate_inputs,
)


class TestValidateInputs:
    """Tests for input validation."""
    
    def test_valid_inputs(self):
        """Test that valid inputs do not raise errors."""
        validate_inputs(0.613, 401, 0.01, "test")
    
    def test_invalid_base_fluid_conductivity(self):
        """Test that negative base fluid conductivity raises ValueError."""
        with pytest.raises(ValueError, match="Base fluid thermal conductivity"):
            validate_inputs(-0.5, 401, 0.01)
    
    def test_invalid_nanoparticle_conductivity(self):
        """Test that negative nanoparticle conductivity raises ValueError."""
        with pytest.raises(ValueError, match="Nanoparticle thermal conductivity"):
            validate_inputs(0.613, -100, 0.01)
    
    def test_invalid_volume_fraction_negative(self):
        """Test that negative volume fraction raises ValueError."""
        with pytest.raises(ValueError, match="Volume fraction"):
            validate_inputs(0.613, 401, -0.1)
    
    def test_invalid_volume_fraction_too_high(self):
        """Test that volume fraction > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Volume fraction"):
            validate_inputs(0.613, 401, 1.5)


class TestMaxwellModel:
    """Tests for Maxwell model."""
    
    def test_zero_volume_fraction(self):
        """Test that zero volume fraction returns base fluid conductivity."""
        k_bf = 0.613
        k_np = 401
        result = maxwell_model(k_bf, k_np, 0)
        assert abs(result - k_bf) < 1e-10
    
    def test_positive_enhancement(self):
        """Test that adding high-k nanoparticles increases conductivity."""
        k_bf = 0.613
        k_np = 401  # Copper
        phi = 0.01
        result = maxwell_model(k_bf, k_np, phi)
        assert result > k_bf
    
    def test_typical_water_copper(self):
        """Test typical water-copper nanofluid calculation."""
        k_bf = 0.613  # Water at 25°C
        k_np = 401  # Copper
        phi = 0.01  # 1% volume fraction
        result = maxwell_model(k_bf, k_np, phi)
        # Expected enhancement ~3% for 1% Cu in water
        enhancement = (result - k_bf) / k_bf * 100
        assert 2 < enhancement < 5


class TestHamiltonCrosserModel:
    """Tests for Hamilton-Crosser model."""
    
    def test_spherical_same_as_maxwell(self):
        """Test that spherical particles give same result as Maxwell."""
        k_bf = 0.613
        k_np = 401
        phi = 0.01
        maxwell_result = maxwell_model(k_bf, k_np, phi)
        hc_result = hamilton_crosser_model(k_bf, k_np, phi, sphericity=1.0)
        assert abs(maxwell_result - hc_result) < 1e-10
    
    def test_non_spherical_enhancement(self):
        """Test that non-spherical particles give higher enhancement."""
        k_bf = 0.613
        k_np = 401
        phi = 0.01
        spherical = hamilton_crosser_model(k_bf, k_np, phi, sphericity=1.0)
        cylindrical = hamilton_crosser_model(k_bf, k_np, phi, sphericity=0.5)
        # Non-spherical should give higher conductivity
        assert cylindrical > spherical
    
    def test_invalid_sphericity(self):
        """Test that invalid sphericity raises ValueError."""
        with pytest.raises(ValueError, match="Sphericity"):
            hamilton_crosser_model(0.613, 401, 0.01, sphericity=0)
        with pytest.raises(ValueError, match="Sphericity"):
            hamilton_crosser_model(0.613, 401, 0.01, sphericity=1.5)


class TestBruggemanModel:
    """Tests for Bruggeman model."""
    
    def test_zero_volume_fraction(self):
        """Test that zero volume fraction returns base fluid conductivity."""
        k_bf = 0.613
        k_np = 401
        result = bruggeman_model(k_bf, k_np, 0)
        assert abs(result - k_bf) < 1e-10
    
    def test_positive_enhancement(self):
        """Test that Bruggeman gives positive enhancement."""
        k_bf = 0.613
        k_np = 401
        phi = 0.05
        result = bruggeman_model(k_bf, k_np, phi)
        assert result > k_bf
    
    def test_higher_phi_higher_enhancement(self):
        """Test that higher volume fraction gives higher conductivity."""
        k_bf = 0.613
        k_np = 401
        result_low = bruggeman_model(k_bf, k_np, 0.01)
        result_high = bruggeman_model(k_bf, k_np, 0.05)
        assert result_high > result_low


class TestYuChoiModel:
    """Tests for Yu-Choi model."""
    
    def test_zero_volume_fraction(self):
        """Test that zero volume fraction returns base fluid conductivity."""
        k_bf = 0.613
        k_np = 401
        result = yu_choi_model(k_bf, k_np, 0)
        assert abs(result - k_bf) < 1e-10
    
    def test_nanolayer_effect(self):
        """Test that nanolayer increases enhancement."""
        k_bf = 0.613
        k_np = 401
        phi = 0.01
        radius = 25
        # No nanolayer vs with nanolayer
        no_layer = yu_choi_model(k_bf, k_np, phi, layer_thickness=0, particle_radius=radius)
        with_layer = yu_choi_model(k_bf, k_np, phi, layer_thickness=2, particle_radius=radius)
        assert with_layer > no_layer
    
    def test_invalid_layer_thickness(self):
        """Test that negative layer thickness raises ValueError."""
        with pytest.raises(ValueError, match="Layer thickness"):
            yu_choi_model(0.613, 401, 0.01, layer_thickness=-1)
    
    def test_invalid_particle_radius(self):
        """Test that non-positive particle radius raises ValueError."""
        with pytest.raises(ValueError, match="Particle radius"):
            yu_choi_model(0.613, 401, 0.01, particle_radius=0)


class TestWaspModel:
    """Tests for Wasp model."""
    
    def test_zero_volume_fraction(self):
        """Test that zero volume fraction returns base fluid conductivity."""
        k_bf = 0.613
        k_np = 401
        result = wasp_model(k_bf, k_np, 0)
        assert abs(result - k_bf) < 1e-10
    
    def test_same_as_maxwell_for_spheres(self):
        """Test that Wasp model equals Maxwell for spherical particles."""
        k_bf = 0.613
        k_np = 401
        phi = 0.01
        wasp_result = wasp_model(k_bf, k_np, phi)
        maxwell_result = maxwell_model(k_bf, k_np, phi)
        assert abs(wasp_result - maxwell_result) < 1e-10


class TestPakChoCorrelation:
    """Tests for Pak-Cho empirical correlation."""
    
    def test_zero_volume_fraction(self):
        """Test that zero volume fraction returns base fluid conductivity."""
        k_bf = 0.613
        result = pak_cho_correlation(k_bf, 0, "Al2O3")
        assert abs(result - k_bf) < 1e-10
    
    def test_known_nanoparticle_types(self):
        """Test that known nanoparticle types work correctly."""
        k_bf = 0.613
        phi = 0.01
        for np_type in ["Al2O3", "TiO2", "CuO", "Cu", "Ag"]:
            result = pak_cho_correlation(k_bf, phi, np_type)
            assert result > k_bf
    
    def test_unknown_nanoparticle_type(self):
        """Test that unknown nanoparticle type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown nanoparticle type"):
            pak_cho_correlation(0.613, 0.01, "UnknownMaterial")
    
    def test_case_insensitive(self):
        """Test that nanoparticle type is case insensitive."""
        k_bf = 0.613
        phi = 0.01
        result_lower = pak_cho_correlation(k_bf, phi, "al2o3")
        result_upper = pak_cho_correlation(k_bf, phi, "AL2O3")
        assert abs(result_lower - result_upper) < 1e-10


class TestModelComparisons:
    """Tests comparing different models."""
    
    def test_all_models_agree_at_zero_phi(self):
        """Test that all models return k_bf when phi=0."""
        k_bf = 0.613
        k_np = 401
        phi = 0
        
        results = [
            maxwell_model(k_bf, k_np, phi),
            hamilton_crosser_model(k_bf, k_np, phi),
            bruggeman_model(k_bf, k_np, phi),
            yu_choi_model(k_bf, k_np, phi),
            wasp_model(k_bf, k_np, phi),
        ]
        
        for result in results:
            assert abs(result - k_bf) < 1e-10
    
    def test_all_models_positive_enhancement(self):
        """Test that all models give positive enhancement for high-k particles."""
        k_bf = 0.613
        k_np = 401  # Copper (high conductivity)
        phi = 0.03
        
        results = [
            maxwell_model(k_bf, k_np, phi),
            hamilton_crosser_model(k_bf, k_np, phi),
            bruggeman_model(k_bf, k_np, phi),
            yu_choi_model(k_bf, k_np, phi),
            wasp_model(k_bf, k_np, phi),
        ]
        
        for result in results:
            assert result > k_bf
