"""
Unit tests for NanofluidSimulator class
"""

import pytest
import json
from nanofluid_simulator import NanofluidSimulator
from nanofluid_simulator.nanoparticles import NanoparticleDatabase


class TestNanofluidSimulatorConfiguration:
    """Tests for simulator configuration."""
    
    def test_set_base_fluid_by_name(self):
        """Test setting base fluid by name from database."""
        sim = NanofluidSimulator()
        sim.set_base_fluid("water")
        config = sim.get_configuration()
        assert config["base_fluid"] == "water"
        assert config["k_base_fluid"] == 0.613
    
    def test_set_base_fluid_by_value(self):
        """Test setting base fluid by thermal conductivity value."""
        sim = NanofluidSimulator()
        sim.set_base_fluid(0.5)
        config = sim.get_configuration()
        assert config["base_fluid"] == "custom"
        assert config["k_base_fluid"] == 0.5
    
    def test_set_nanoparticle_by_name(self):
        """Test setting nanoparticle by formula from database."""
        sim = NanofluidSimulator()
        sim.set_nanoparticle("Cu")
        config = sim.get_configuration()
        assert config["nanoparticle"] == "Cu"
        assert config["k_nanoparticle"] == 401
    
    def test_set_nanoparticle_by_value(self):
        """Test setting nanoparticle by thermal conductivity value."""
        sim = NanofluidSimulator()
        sim.set_nanoparticle(500)
        config = sim.get_configuration()
        assert config["nanoparticle"] == "custom"
        assert config["k_nanoparticle"] == 500
    
    def test_set_volume_fraction(self):
        """Test setting volume fraction."""
        sim = NanofluidSimulator()
        sim.set_volume_fraction(0.05)
        config = sim.get_configuration()
        assert config["volume_fraction"] == 0.05
    
    def test_set_volume_fraction_percent(self):
        """Test setting volume fraction as percentage."""
        sim = NanofluidSimulator()
        sim.set_volume_fraction_percent(5)
        config = sim.get_configuration()
        assert abs(config["volume_fraction"] - 0.05) < 1e-10
    
    def test_method_chaining(self):
        """Test that configuration methods support chaining."""
        sim = NanofluidSimulator()
        result = (
            sim.set_base_fluid("water")
            .set_nanoparticle("Cu")
            .set_volume_fraction(0.01)
        )
        assert result is sim
    
    def test_invalid_base_fluid(self):
        """Test that invalid base fluid name raises error."""
        sim = NanofluidSimulator()
        with pytest.raises(KeyError):
            sim.set_base_fluid("unknown_fluid")
    
    def test_invalid_nanoparticle(self):
        """Test that invalid nanoparticle formula raises error."""
        sim = NanofluidSimulator()
        with pytest.raises(KeyError):
            sim.set_nanoparticle("UnknownMaterial")
    
    def test_invalid_volume_fraction(self):
        """Test that invalid volume fraction raises error."""
        sim = NanofluidSimulator()
        with pytest.raises(ValueError):
            sim.set_volume_fraction(-0.1)
        with pytest.raises(ValueError):
            sim.set_volume_fraction(1.5)


class TestNanofluidSimulatorCalculations:
    """Tests for simulator calculations."""
    
    @pytest.fixture
    def configured_sim(self):
        """Create a configured simulator for testing."""
        sim = NanofluidSimulator()
        sim.set_base_fluid("water")
        sim.set_nanoparticle("Cu")
        sim.set_volume_fraction(0.01)
        return sim
    
    def test_calculate_maxwell(self, configured_sim):
        """Test Maxwell model calculation."""
        result = configured_sim.calculate_maxwell()
        assert result.model_name == "Maxwell"
        assert result.k_effective > result.k_base_fluid
        assert result.enhancement_percent > 0
    
    def test_calculate_hamilton_crosser(self, configured_sim):
        """Test Hamilton-Crosser model calculation."""
        result = configured_sim.calculate_hamilton_crosser()
        assert result.model_name == "Hamilton-Crosser"
        assert result.k_effective > result.k_base_fluid
    
    def test_calculate_bruggeman(self, configured_sim):
        """Test Bruggeman model calculation."""
        result = configured_sim.calculate_bruggeman()
        assert result.model_name == "Bruggeman"
        assert result.k_effective > result.k_base_fluid
    
    def test_calculate_yu_choi(self, configured_sim):
        """Test Yu-Choi model calculation."""
        result = configured_sim.calculate_yu_choi()
        assert result.model_name == "Yu-Choi"
        assert result.k_effective > result.k_base_fluid
    
    def test_calculate_wasp(self, configured_sim):
        """Test Wasp model calculation."""
        result = configured_sim.calculate_wasp()
        assert result.model_name == "Wasp"
        assert result.k_effective > result.k_base_fluid
    
    def test_calculate_pak_cho(self, configured_sim):
        """Test Pak-Cho correlation."""
        result = configured_sim.calculate_pak_cho()
        assert result.model_name == "Pak-Cho"
        assert result.k_effective > result.k_base_fluid
    
    def test_calculate_all_models(self, configured_sim):
        """Test calculating all models at once."""
        results = configured_sim.calculate_all_models()
        assert len(results) >= 5
        model_names = [r.model_name for r in results]
        assert "Maxwell" in model_names
        assert "Bruggeman" in model_names
    
    def test_unconfigured_simulator_raises_error(self):
        """Test that unconfigured simulator raises error."""
        sim = NanofluidSimulator()
        with pytest.raises(ValueError, match="Base fluid not set"):
            sim.calculate_maxwell()
        
        sim.set_base_fluid("water")
        with pytest.raises(ValueError, match="Nanoparticle not set"):
            sim.calculate_maxwell()


class TestNanofluidSimulatorOutput:
    """Tests for simulator output methods."""
    
    @pytest.fixture
    def configured_sim(self):
        """Create a configured simulator for testing."""
        sim = NanofluidSimulator()
        sim.set_base_fluid("water")
        sim.set_nanoparticle("Al2O3")
        sim.set_volume_fraction(0.02)
        return sim
    
    def test_compare_models_output(self, configured_sim):
        """Test that compare_models produces formatted output."""
        output = configured_sim.compare_models()
        assert "Maxwell" in output
        assert "Hamilton-Crosser" in output
        assert "Water" in output or "water" in output
        assert "Al2O3" in output or "Al₂O₃" in output
    
    def test_to_json_output(self, configured_sim):
        """Test that to_json produces valid JSON."""
        json_str = configured_sim.to_json()
        data = json.loads(json_str)
        assert "configuration" in data
        assert "results" in data
        assert len(data["results"]) >= 5
    
    def test_str_representation(self, configured_sim):
        """Test string representation of simulator."""
        str_repr = str(configured_sim)
        assert "NanofluidSimulator" in str_repr
        assert "water" in str_repr


class TestParametricStudy:
    """Tests for parametric study functionality."""
    
    def test_parametric_study_basic(self):
        """Test basic parametric study."""
        sim = NanofluidSimulator()
        sim.set_base_fluid("water")
        sim.set_nanoparticle("Cu")
        
        volume_fractions = [0.01, 0.02, 0.03]
        results = sim.parametric_study(
            volume_fractions,
            models=["maxwell", "bruggeman"]
        )
        
        assert "maxwell" in results
        assert "bruggeman" in results
        assert len(results["maxwell"]) == 3
    
    def test_parametric_study_increasing_enhancement(self):
        """Test that enhancement increases with volume fraction."""
        sim = NanofluidSimulator()
        sim.set_base_fluid("water")
        sim.set_nanoparticle("Cu")
        
        volume_fractions = [0.01, 0.02, 0.03, 0.04, 0.05]
        results = sim.parametric_study(
            volume_fractions,
            models=["maxwell"]
        )
        
        maxwell_results = results["maxwell"]
        for i in range(1, len(maxwell_results)):
            assert maxwell_results[i].k_effective > maxwell_results[i-1].k_effective


class TestNanoparticleDatabase:
    """Tests for NanoparticleDatabase."""
    
    def test_get_nanoparticle(self):
        """Test retrieving nanoparticle properties."""
        cu = NanoparticleDatabase.get_nanoparticle("Cu")
        assert cu.thermal_conductivity == 401
        assert cu.name == "Copper"
    
    def test_get_base_fluid(self):
        """Test retrieving base fluid properties."""
        water = NanoparticleDatabase.get_base_fluid("water")
        assert water.thermal_conductivity == 0.613
        assert water.name == "Water"
    
    def test_list_nanoparticles(self):
        """Test listing available nanoparticles."""
        np_list = NanoparticleDatabase.list_nanoparticles()
        assert "Cu" in np_list
        assert "Al2O3" in np_list
        assert "CNT" in np_list
    
    def test_list_base_fluids(self):
        """Test listing available base fluids."""
        fluid_list = NanoparticleDatabase.list_base_fluids()
        assert "water" in fluid_list
        assert "ethylene_glycol" in fluid_list
    
    def test_case_insensitive_nanoparticle(self):
        """Test that nanoparticle lookup is case insensitive."""
        cu_lower = NanoparticleDatabase.get_nanoparticle("cu")
        cu_upper = NanoparticleDatabase.get_nanoparticle("CU")
        assert cu_lower.thermal_conductivity == cu_upper.thermal_conductivity


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        sim = NanofluidSimulator()
        sim.set_base_fluid("water")
        sim.set_nanoparticle("Cu")
        sim.set_volume_fraction(0.01)
        
        result = sim.calculate_maxwell()
        result_dict = result.to_dict()
        
        assert "model" in result_dict
        assert "k_effective" in result_dict
        assert "enhancement_percent" in result_dict
    
    def test_result_str_representation(self):
        """Test string representation of result."""
        sim = NanofluidSimulator()
        sim.set_base_fluid("water")
        sim.set_nanoparticle("Cu")
        sim.set_volume_fraction(0.01)
        
        result = sim.calculate_maxwell()
        str_repr = str(result)
        
        assert "Maxwell" in str_repr
        assert "W/m·K" in str_repr
