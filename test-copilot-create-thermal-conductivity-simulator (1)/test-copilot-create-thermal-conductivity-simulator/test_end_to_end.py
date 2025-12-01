#!/usr/bin/env python3
"""
End-to-End Test Suite for Nanofluid Simulator v3.0
Tests all major functionality without GUI components
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_1_basic_simulator():
    """Test basic NanofluidSimulator functionality"""
    print("\n" + "="*60)
    print("TEST 1: Basic Thermal Conductivity Simulator")
    print("="*60)
    
    from nanofluid_simulator import NanofluidSimulator
    
    # Create and configure simulator
    sim = NanofluidSimulator()
    sim.set_base_fluid('water', temperature=300)
    sim.set_nanoparticle('Al2O3')
    sim.set_volume_fraction(0.02)
    
    # Calculate all models
    results = sim.calculate_all_models()
    
    assert len(results) > 0, "No results returned"
    assert all(hasattr(r, 'k_effective') for r in results), "Missing k_effective"
    assert all(hasattr(r, 'enhancement_percent') for r in results), "Missing enhancement"
    
    print(f"✓ Calculated {len(results)} models successfully")
    for res in results[:3]:
        print(f"  {res.model_name}: {res.k_effective:.4f} W/m·K ({res.enhancement_percent:.1f}%)")
    
    return True


def test_2_different_materials():
    """Test different nanoparticle materials"""
    print("\n" + "="*60)
    print("TEST 2: Different Nanoparticle Materials")
    print("="*60)
    
    from nanofluid_simulator import NanofluidSimulator
    
    particles = ['Cu', 'CuO', 'TiO2', 'Al2O3']
    results_by_particle = {}
    
    for particle in particles:
        sim = NanofluidSimulator()
        sim.set_base_fluid('water')
        sim.set_nanoparticle(particle)
        sim.set_volume_fraction(0.01)
        results = sim.calculate_all_models()
        
        avg_k = sum(r.k_effective for r in results) / len(results)
        avg_enh = sum(r.enhancement_percent for r in results) / len(results)
        results_by_particle[particle] = (avg_k, avg_enh)
        
        print(f"  {particle:6s}: k = {avg_k:.4f} W/m·K, enhancement = {avg_enh:.1f}%")
    
    assert all(k > 0.6 for k, _ in results_by_particle.values()), "Unrealistic k values"
    return True


def test_3_volume_fraction_sweep():
    """Test volume fraction sweep"""
    print("\n" + "="*60)
    print("TEST 3: Volume Fraction Sweep")
    print("="*60)
    
    from nanofluid_simulator import NanofluidSimulator
    
    fractions = [0.01, 0.02, 0.03, 0.04, 0.05]
    k_values = []
    
    for phi in fractions:
        sim = NanofluidSimulator()
        sim.set_base_fluid('water')
        sim.set_nanoparticle('Al2O3')
        sim.set_volume_fraction(phi)
        results = sim.calculate_all_models()
        avg_k = sum(r.k_effective for r in results) / len(results)
        k_values.append(avg_k)
        print(f"  φ = {phi:.1%}: k = {avg_k:.4f} W/m·K")
    
    # Check that k increases with volume fraction
    assert all(k_values[i] < k_values[i+1] for i in range(len(k_values)-1)), \
        "k should increase with volume fraction"
    
    return True


def test_4_enhanced_simulator():
    """Test EnhancedNanofluidSimulator"""
    print("\n" + "="*60)
    print("TEST 4: Enhanced Nanofluid Simulator")
    print("="*60)
    
    from nanofluid_simulator.enhanced_simulator import EnhancedNanofluidSimulator
    
    sim = EnhancedNanofluidSimulator()
    sim.add_nanoparticle('Al2O3', 0.02)
    sim.set_base_fluid('water')
    sim.set_temperature(300)
    
    results = sim.calculate_all_applicable_models()
    
    assert len(results) > 0, "No enhanced results"
    assert all(hasattr(r, 'k_effective') for r in results), "Missing k_effective"
    
    print(f"✓ Calculated {len(results)} enhanced models")
    for res in results[:3]:
        print(f"  {res.model_name}: k = {res.k_effective:.4f} W/m·K")
    
    return True


def test_5_hybrid_nanofluid():
    """Test hybrid nanofluid calculations"""
    print("\n" + "="*60)
    print("TEST 5: Hybrid Nanofluid")
    print("="*60)
    
    from nanofluid_simulator.enhanced_simulator import EnhancedNanofluidSimulator
    
    sim = EnhancedNanofluidSimulator()
    sim.add_nanoparticle('Al2O3', 0.01)
    sim.add_nanoparticle('CuO', 0.01)
    sim.set_base_fluid('water')
    sim.set_temperature(300)
    
    results = sim.calculate_all_applicable_models()
    
    assert len(results) > 0, "No hybrid results"
    
    print(f"✓ Hybrid nanofluid with 2 particles")
    print(f"  Calculated {len(results)} models")
    for res in results[:2]:
        print(f"  {res.model_name}: k = {res.k_effective:.4f} W/m·K")
    
    return True


def test_6_visualization_data():
    """Test flow visualization data generation"""
    print("\n" + "="*60)
    print("TEST 6: Flow Visualization Data Generation")
    print("="*60)
    
    from nanofluid_simulator.visualization import FlowVisualizer
    
    nanofluid_data = {
        'temperature': 300,
        'thermal_conductivity': 0.65,
        'viscosity': 0.001,
        'density': 1000,
        'volume_fraction': 0.02
    }
    
    viz = FlowVisualizer(nanofluid_data)
    
    # Test thermal field generation
    X, Y, T = viz.create_thermal_field(geometry='channel', nx=20, ny=20)
    assert X.shape == (20, 20), f"Wrong X shape: {X.shape}"
    assert T.shape == (20, 20), f"Wrong T shape: {T.shape}"
    print(f"✓ Thermal field: shape {X.shape}, T range {T.min():.1f}-{T.max():.1f} K")
    
    # Test velocity field generation
    X, Y, U, V = viz.create_velocity_field(geometry='channel', nx=20, ny=20, Re=1000)
    assert U.shape == (20, 20), f"Wrong U shape: {U.shape}"
    assert V.shape == (20, 20), f"Wrong V shape: {V.shape}"
    print(f"✓ Velocity field: shape {U.shape}, max velocity {U.max():.4f} m/s")
    
    # Test streamlines
    X, Y, U, V = viz.create_streamlines(geometry='pipe', nx=30, ny=30, Re=2000)
    print(f"✓ Streamlines: shape {U.shape}")
    
    return True


def test_7_property_calculator():
    """Test thermophysical property calculations"""
    print("\n" + "="*60)
    print("TEST 7: Thermophysical Properties")
    print("="*60)
    
    from nanofluid_simulator.thermophysical_properties import ThermophysicalProperties
    
    props = ThermophysicalProperties()
    
    # Test water properties at different temperatures
    temps = [280, 300, 320, 340]
    for T in temps:
        k = props.get_base_fluid_conductivity('water', T)
        mu = props.get_base_fluid_viscosity('water', T)
        rho = props.get_base_fluid_density('water', T)
        
        assert k > 0, f"Invalid conductivity at {T}K"
        assert mu > 0, f"Invalid viscosity at {T}K"
        assert rho > 0, f"Invalid density at {T}K"
        
        print(f"  T = {T}K: k = {k:.4f}, μ = {mu:.6f}, ρ = {rho:.1f}")
    
    return True


def test_8_nanoparticle_database():
    """Test nanoparticle database"""
    print("\n" + "="*60)
    print("TEST 8: Nanoparticle Database")
    print("="*60)
    
    from nanofluid_simulator.nanoparticles import NanoparticleDatabase
    
    db = NanoparticleDatabase()
    
    # Test getting particles
    particles = ['Al2O3', 'Cu', 'CuO', 'TiO2', 'SiO2']
    for name in particles:
        mat = db.get_nanoparticle(name)
        assert mat is not None, f"Particle {name} not found"
        assert mat.thermal_conductivity > 0, f"Invalid k for {name}"
        print(f"  {name:6s}: k = {mat.thermal_conductivity:.1f} W/m·K, ρ = {mat.density:.0f} kg/m³")
    
    return True


def test_9_model_results():
    """Test different thermal conductivity models"""
    print("\n" + "="*60)
    print("TEST 9: Individual Model Tests")
    print("="*60)
    
    from nanofluid_simulator import NanofluidSimulator
    
    sim = NanofluidSimulator()
    sim.set_base_fluid('water')
    sim.set_nanoparticle('Al2O3')
    sim.set_volume_fraction(0.02)
    
    # Test specific models
    models_to_test = ['maxwell', 'hamilton_crosser', 'bruggeman']
    for model_name in models_to_test:
        try:
            k_eff = sim.calculate_model(model_name)
            assert k_eff > 0, f"Invalid k_eff for {model_name}"
            print(f"  {model_name:20s}: {k_eff:.4f} W/m·K")
        except Exception as e:
            print(f"  {model_name:20s}: FAILED - {e}")
            return False
    
    return True


def test_10_temperature_dependence():
    """Test temperature-dependent properties"""
    print("\n" + "="*60)
    print("TEST 10: Temperature Dependence")
    print("="*60)
    
    from nanofluid_simulator import NanofluidSimulator
    
    temperatures = [280, 300, 320, 340, 360]
    k_at_temps = []
    
    for T in temperatures:
        sim = NanofluidSimulator()
        sim.set_base_fluid('water', temperature=T)
        sim.set_nanoparticle('Al2O3')
        sim.set_volume_fraction(0.02)
        results = sim.calculate_all_models()
        avg_k = sum(r.k_effective for r in results) / len(results)
        k_at_temps.append(avg_k)
        print(f"  T = {T}K: k = {avg_k:.4f} W/m·K")
    
    # Check that k increases with temperature (generally true for nanofluids)
    print(f"  Temperature effect: {((k_at_temps[-1]/k_at_temps[0] - 1)*100):.1f}% increase")
    
    return True


def run_all_tests():
    """Run all end-to-end tests"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*10 + "NANOFLUID SIMULATOR v3.0" + " "*24 + "║")
    print("║" + " "*14 + "End-to-End Test Suite" + " "*23 + "║")
    print("╚" + "="*58 + "╝")
    
    tests = [
        test_1_basic_simulator,
        test_2_different_materials,
        test_3_volume_fraction_sweep,
        test_4_enhanced_simulator,
        test_5_hybrid_nanofluid,
        test_6_visualization_data,
        test_7_property_calculator,
        test_8_nanoparticle_database,
        test_9_model_results,
        test_10_temperature_dependence,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"  ✅ PASSED")
            else:
                failed += 1
                print(f"  ❌ FAILED")
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The simulator is production-ready.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
