"""
Baseline Test Suite for LoRA Format Resolution

Tests all 8 LoRA input formats and their code paths through the current system.
This establishes a baseline before refactoring, so we can validate that the
new unified resolver produces identical results.

Run with: python tests/test_lora_formats_baseline.py
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_MODEL_TYPE = "vace_14B_cocktail_2_2"
TEST_TASK_ID = "test_baseline"


class TestResult:
    """Track test results for reporting."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record_pass(self, test_name: str):
        self.passed += 1
        print(f"  ✅ {test_name}")
    
    def record_fail(self, test_name: str, reason: str):
        self.failed += 1
        self.errors.append((test_name, reason))
        print(f"  ❌ {test_name}: {reason}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailures:")
            for test_name, reason in self.errors:
                print(f"  - {test_name}: {reason}")
        print(f"{'='*70}\n")
        return self.failed == 0


def dprint(msg: str):
    """Debug print with prefix."""
    print(f"    [DEBUG] {msg}")


# =============================================================================
# Format Processing Functions (Current System)
# =============================================================================

def process_format_1_wgp_legacy(params: dict) -> dict:
    """
    Test Format 1: WGP Legacy (activated_loras list + loras_multipliers string)
    
    Current code path: worker.py lines 1022-1042
    """
    print("  📋 Processing Format 1: WGP Legacy")
    result = params.copy()
    
    # Simulate worker.py conversion
    if "activated_loras" in result:
        loras = result["activated_loras"]
        if isinstance(loras, str):
            result["lora_names"] = [lora.strip() for lora in loras.split(",") if lora.strip()]
        elif isinstance(loras, list):
            result["lora_names"] = loras
        del result["activated_loras"]
    
    if "loras_multipliers" in result:
        multipliers = result["loras_multipliers"]
        if isinstance(multipliers, str):
            if ";" in multipliers:
                # Phase-config format (space or comma separated)
                # Check which separator is primary (whichever appears without the other)
                if "," in multipliers and " " not in multipliers.replace(",", ""):
                    sep = ","
                else:
                    sep = " "
                result["lora_multipliers"] = [x.strip() for x in multipliers.split(sep) if x.strip()]
            else:
                # Regular format (space or comma separated)
                # Try comma first, fall back to space
                if "," in multipliers:
                    result["lora_multipliers"] = [float(x.strip()) for x in multipliers.split(",") if x.strip()]
                else:
                    result["lora_multipliers"] = [float(x.strip()) for x in multipliers.split() if x.strip()]
    
    dprint(f"Result: lora_names={result.get('lora_names', [])}, lora_multipliers={result.get('lora_multipliers', [])}")
    return result


def process_format_2_csv_string(params: dict) -> dict:
    """
    Test Format 2: CSV String (comma-separated strings)
    
    Current code path: worker.py lines 1022-1042 (same as Format 1)
    """
    print("  📋 Processing Format 2: CSV String")
    return process_format_1_wgp_legacy(params)


def process_format_3_internal_list(params: dict) -> dict:
    """
    Test Format 3: Internal List (lora_names + lora_multipliers as lists)
    
    Current code path: Already in correct format, but may need normalization
    """
    print("  📋 Processing Format 3: Internal List")
    result = params.copy()
    
    # Simulate lora_utils.py normalization
    if "lora_multipliers" in result:
        multipliers = result["lora_multipliers"]
        if isinstance(multipliers, list):
            # Check if phase-config
            is_phase_config = any(";" in str(m) for m in multipliers)
            if not is_phase_config:
                # Convert to floats
                try:
                    result["lora_multipliers"] = [float(m) for m in multipliers]
                except (ValueError, TypeError):
                    pass  # Keep as-is
    
    dprint(f"Result: lora_names={result.get('lora_names', [])}, lora_multipliers={result.get('lora_multipliers', [])}")
    return result


def process_format_4_phase_config(params: dict) -> dict:
    """
    Test Format 4: Phase-Config (multi-phase guidance strings)
    
    Current code path: worker.py parse_phase_config() lines 693-775
    """
    print("  📋 Processing Format 4: Phase-Config")
    result = params.copy()
    
    # Phase-config multipliers should be preserved as strings
    if "lora_multipliers" in result:
        multipliers = result["lora_multipliers"]
        if isinstance(multipliers, list):
            # Check if phase-config format
            is_phase_config = any(";" in str(m) for m in multipliers)
            if is_phase_config:
                dprint(f"Detected phase-config format: {multipliers}")
    
    dprint(f"Result: lora_names={result.get('lora_names', [])}, lora_multipliers={result.get('lora_multipliers', [])}")
    return result


def process_format_5_additional_dict(params: dict) -> dict:
    """
    Test Format 5: Additional LoRAs Dict (URL → multiplier map)
    
    Current code path: lora_utils.py normalize_lora_format() lines 420-448
    """
    print("  📋 Processing Format 5: Additional Dict")
    result = params.copy()
    
    additional = result.get("additional_loras", {})
    if additional and isinstance(additional, dict):
        current_loras = result.get("lora_names", [])
        current_mults = result.get("lora_multipliers", [])
        
        for lora_url, multiplier in additional.items():
            # Extract filename from URL
            if lora_url.startswith("http"):
                filename = Path(lora_url).name
            else:
                filename = lora_url
            
            if filename not in current_loras:
                current_loras.append(filename)
                current_mults.append(multiplier)
        
        result["lora_names"] = current_loras
        result["lora_multipliers"] = current_mults
        del result["additional_loras"]
    
    dprint(f"Result: lora_names={result.get('lora_names', [])}, lora_multipliers={result.get('lora_multipliers', [])}")
    return result


def process_format_6_qwen_task(params: dict) -> dict:
    """
    Test Format 6: Qwen Task Format (list of dicts with path/scale)
    
    Current code path: worker.py Qwen handlers lines 1153-1161
    """
    print("  📋 Processing Format 6: Qwen Task")
    result = params.copy()
    
    qwen_loras = result.get("loras", [])
    if isinstance(qwen_loras, list):
        lora_names = result.get("lora_names", [])
        lora_mults = result.get("lora_multipliers", [])
        
        for lora_dict in qwen_loras:
            if isinstance(lora_dict, dict):
                lora_path = lora_dict.get("path") or lora_dict.get("url")
                lora_scale = lora_dict.get("scale") or lora_dict.get("strength", 1.0)
                
                if lora_path:
                    lora_filename = Path(lora_path).name
                    if lora_filename not in lora_names:
                        lora_names.append(lora_filename)
                        lora_mults.append(float(lora_scale))
        
        result["lora_names"] = lora_names
        result["lora_multipliers"] = lora_mults
    
    dprint(f"Result: lora_names={result.get('lora_names', [])}, lora_multipliers={result.get('lora_multipliers', [])}")
    return result


def process_format_7_model_preset(params: dict, model_type: str) -> dict:
    """
    Test Format 7: Model Preset (from JSON model definition)
    
    Current code path: headless_wgp.py _resolve_parameters() lines 744-792
    Note: This requires loading model JSON, so we'll simulate with fake data
    """
    print("  📋 Processing Format 7: Model Preset (simulated)")
    result = params.copy()
    
    # Simulate model preset having default LoRAs
    if "2_2" in model_type or "cocktail" in model_type:
        # Wan 2.2 models have built-in acceleration LoRAs
        if "lora_names" not in result:
            result["lora_names"] = ["CausVid", "DetailEnhancerV1"]
            result["lora_multipliers"] = [1.0, 0.2]
            dprint("Applied default Wan 2.2 acceleration LoRAs")
    
    dprint(f"Result: lora_names={result.get('lora_names', [])}, lora_multipliers={result.get('lora_multipliers', [])}")
    return result


def process_format_8_feature_flags(params: dict, model_type: str) -> dict:
    """
    Test Format 8: Boolean Feature Flags
    
    Current code path: lora_utils.py detect_lora_optimization_flags()
    """
    print("  📋 Processing Format 8: Feature Flags")
    result = params.copy()
    
    lora_names = result.get("lora_names", [])
    lora_mults = result.get("lora_multipliers", [])
    
    # CausVid flag
    if result.get("use_causvid_lora"):
        causvid_lora = "CausVid" if "2_2" in model_type else "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"
        if causvid_lora not in lora_names:
            lora_names.append(causvid_lora)
            lora_mults.append(1.0)
            dprint(f"Added CausVid LoRA: {causvid_lora}")
    
    # LightI2X flag
    if result.get("use_lighti2x_lora"):
        lighti2x_lora = "high_noise_model.safetensors"
        if lighti2x_lora not in lora_names:
            lora_names.append(lighti2x_lora)
            lora_mults.append(1.0)
            dprint(f"Added LightI2X LoRA: {lighti2x_lora}")
    
    # Reward LoRA flag
    if result.get("apply_reward_lora"):
        reward_lora = "reward_model.safetensors"
        if reward_lora not in lora_names:
            lora_names.append(reward_lora)
            lora_mults.append(0.5)
            dprint(f"Added reward LoRA: {reward_lora}")
    
    result["lora_names"] = lora_names
    result["lora_multipliers"] = lora_mults
    
    dprint(f"Result: lora_names={result.get('lora_names', [])}, lora_multipliers={result.get('lora_multipliers', [])}")
    return result


# =============================================================================
# Test Cases
# =============================================================================

def test_format_1_wgp_legacy():
    """Test Format 1: WGP Legacy (activated_loras list + loras_multipliers string)"""
    print("\n" + "="*70)
    print("TEST: Format 1 - WGP Legacy")
    print("="*70)
    
    results = TestResult()
    
    # Test 1a: List + space-separated string
    print("\nTest 1a: List + space-separated multipliers")
    params = {
        "activated_loras": ["lora1.safetensors", "lora2.safetensors"],
        "loras_multipliers": "1.0 0.8"
    }
    result = process_format_1_wgp_legacy(params)
    
    if result.get("lora_names") == ["lora1.safetensors", "lora2.safetensors"]:
        results.record_pass("List conversion")
    else:
        results.record_fail("List conversion", f"Expected ['lora1.safetensors', 'lora2.safetensors'], got {result.get('lora_names')}")
    
    if result.get("lora_multipliers") == [1.0, 0.8]:
        results.record_pass("Multiplier parsing (space-separated)")
    else:
        results.record_fail("Multiplier parsing", f"Expected [1.0, 0.8], got {result.get('lora_multipliers')}")
    
    # Test 1b: CSV string + comma-separated multipliers
    print("\nTest 1b: CSV string + comma-separated multipliers")
    params = {
        "activated_loras": "lora1.safetensors,lora2.safetensors",
        "loras_multipliers": "1.0,0.8"
    }
    result = process_format_1_wgp_legacy(params)
    
    if result.get("lora_names") == ["lora1.safetensors", "lora2.safetensors"]:
        results.record_pass("CSV conversion")
    else:
        results.record_fail("CSV conversion", f"Got {result.get('lora_names')}")
    
    # Test 1c: Phase-config multipliers
    print("\nTest 1c: Phase-config multipliers (space-separated)")
    params = {
        "activated_loras": ["lora1.safetensors"],
        "loras_multipliers": "1.0;0;0 0;0.8;1.0"
    }
    result = process_format_1_wgp_legacy(params)
    
    if result.get("lora_multipliers") == ["1.0;0;0", "0;0.8;1.0"]:
        results.record_pass("Phase-config parsing")
    else:
        results.record_fail("Phase-config parsing", f"Expected ['1.0;0;0', '0;0.8;1.0'], got {result.get('lora_multipliers')}")
    
    results.summary()
    assert results.failed == 0, f"{results.failed} sub-test(s) failed"


def test_format_3_internal_list():
    """Test Format 3: Internal List (lora_names + lora_multipliers)"""
    print("\n" + "="*70)
    print("TEST: Format 3 - Internal List")
    print("="*70)
    
    results = TestResult()
    
    # Test 3a: Regular multipliers (should convert to floats)
    print("\nTest 3a: Regular multipliers")
    params = {
        "lora_names": ["lora1.safetensors", "lora2.safetensors"],
        "lora_multipliers": [1.0, 0.8]
    }
    result = process_format_3_internal_list(params)
    
    if result.get("lora_multipliers") == [1.0, 0.8]:
        results.record_pass("Float multipliers preserved")
    else:
        results.record_fail("Float multipliers", f"Got {result.get('lora_multipliers')}")
    
    # Test 3b: Phase-config multipliers (should preserve as strings)
    print("\nTest 3b: Phase-config multipliers")
    params = {
        "lora_names": ["lora1.safetensors"],
        "lora_multipliers": ["1.0;0;0", "0;0.8;1.0"]
    }
    result = process_format_3_internal_list(params)
    
    if result.get("lora_multipliers") == ["1.0;0;0", "0;0.8;1.0"]:
        results.record_pass("Phase-config strings preserved")
    else:
        results.record_fail("Phase-config strings", f"Got {result.get('lora_multipliers')}")
    
    results.summary()
    assert results.failed == 0, f"{results.failed} sub-test(s) failed"


def test_format_5_additional_dict():
    """Test Format 5: Additional LoRAs Dict"""
    print("\n" + "="*70)
    print("TEST: Format 5 - Additional LoRAs Dict")
    print("="*70)
    
    results = TestResult()
    
    # Test 5a: URLs with multipliers
    print("\nTest 5a: URLs with multipliers")
    params = {
        "additional_loras": {
            "https://example.com/lora1.safetensors": 1.0,
            "https://example.com/lora2.safetensors": 0.8
        }
    }
    result = process_format_5_additional_dict(params)
    
    expected_names = ["lora1.safetensors", "lora2.safetensors"]
    if result.get("lora_names") == expected_names:
        results.record_pass("URL filename extraction")
    else:
        results.record_fail("URL filename extraction", f"Expected {expected_names}, got {result.get('lora_names')}")
    
    if result.get("lora_multipliers") == [1.0, 0.8]:
        results.record_pass("Multiplier preservation")
    else:
        results.record_fail("Multiplier preservation", f"Got {result.get('lora_multipliers')}")
    
    # Test 5b: Local filenames
    print("\nTest 5b: Local filenames")
    params = {
        "additional_loras": {
            "lora1.safetensors": 1.0,
            "lora2.safetensors": 0.5
        }
    }
    result = process_format_5_additional_dict(params)
    
    if result.get("lora_names") == ["lora1.safetensors", "lora2.safetensors"]:
        results.record_pass("Local filename handling")
    else:
        results.record_fail("Local filename handling", f"Got {result.get('lora_names')}")
    
    # Test 5c: Merge with existing
    print("\nTest 5c: Merge with existing LoRAs")
    params = {
        "lora_names": ["existing.safetensors"],
        "lora_multipliers": [1.0],
        "additional_loras": {
            "new.safetensors": 0.8
        }
    }
    result = process_format_5_additional_dict(params)
    
    expected_names = ["existing.safetensors", "new.safetensors"]
    if result.get("lora_names") == expected_names:
        results.record_pass("Merge with existing")
    else:
        results.record_fail("Merge with existing", f"Expected {expected_names}, got {result.get('lora_names')}")
    
    results.summary()
    assert results.failed == 0, f"{results.failed} sub-test(s) failed"


def test_format_6_qwen_task():
    """Test Format 6: Qwen Task Format"""
    print("\n" + "="*70)
    print("TEST: Format 6 - Qwen Task Format")
    print("="*70)
    
    results = TestResult()
    
    # Test 6a: path + scale
    print("\nTest 6a: path + scale")
    params = {
        "loras": [
            {"path": "lora1.safetensors", "scale": 1.0},
            {"path": "/full/path/to/lora2.safetensors", "scale": 0.8}
        ]
    }
    result = process_format_6_qwen_task(params)
    
    expected_names = ["lora1.safetensors", "lora2.safetensors"]
    if result.get("lora_names") == expected_names:
        results.record_pass("Path extraction")
    else:
        results.record_fail("Path extraction", f"Expected {expected_names}, got {result.get('lora_names')}")
    
    # Test 6b: url + strength (alternative field names)
    print("\nTest 6b: url + strength")
    params = {
        "loras": [
            {"url": "https://example.com/lora1.safetensors", "strength": 0.9}
        ]
    }
    result = process_format_6_qwen_task(params)
    
    if result.get("lora_names") == ["lora1.safetensors"]:
        results.record_pass("URL extraction with strength field")
    else:
        results.record_fail("URL extraction", f"Got {result.get('lora_names')}")
    
    if result.get("lora_multipliers") == [0.9]:
        results.record_pass("Strength field parsing")
    else:
        results.record_fail("Strength field", f"Got {result.get('lora_multipliers')}")
    
    results.summary()
    assert results.failed == 0, f"{results.failed} sub-test(s) failed"


def test_format_8_feature_flags():
    """Test Format 8: Boolean Feature Flags"""
    print("\n" + "="*70)
    print("TEST: Format 8 - Boolean Feature Flags")
    print("="*70)
    
    results = TestResult()
    
    # Test 8a: CausVid flag (Wan 2.2)
    print("\nTest 8a: CausVid flag (Wan 2.2 model)")
    params = {
        "use_causvid_lora": True
    }
    result = process_format_8_feature_flags(params, "vace_14B_cocktail_2_2")
    
    if "CausVid" in result.get("lora_names", []):
        results.record_pass("CausVid flag → LoRA name (2.2)")
    else:
        results.record_fail("CausVid flag", f"Got {result.get('lora_names')}")
    
    # Test 8b: LightI2X flag
    print("\nTest 8b: LightI2X flag")
    params = {
        "use_lighti2x_lora": True
    }
    result = process_format_8_feature_flags(params, "vace_14B_cocktail_2_2")
    
    if "high_noise_model.safetensors" in result.get("lora_names", []):
        results.record_pass("LightI2X flag → LoRA name")
    else:
        results.record_fail("LightI2X flag", f"Got {result.get('lora_names')}")
    
    # Test 8c: Reward LoRA flag
    print("\nTest 8c: Reward LoRA flag")
    params = {
        "apply_reward_lora": True
    }
    result = process_format_8_feature_flags(params, "vace_14B")
    
    if "reward_model.safetensors" in result.get("lora_names", []):
        results.record_pass("Reward LoRA flag")
    else:
        results.record_fail("Reward LoRA flag", f"Got {result.get('lora_names')}")
    
    if result.get("lora_multipliers", [None])[0] == 0.5:
        results.record_pass("Reward LoRA strength (0.5)")
    else:
        results.record_fail("Reward LoRA strength", f"Got {result.get('lora_multipliers')}")
    
    # Test 8d: Multiple flags
    print("\nTest 8d: Multiple flags combined")
    params = {
        "use_causvid_lora": True,
        "use_lighti2x_lora": True,
        "apply_reward_lora": True
    }
    result = process_format_8_feature_flags(params, "vace_14B_cocktail_2_2")
    
    expected_loras = ["CausVid", "high_noise_model.safetensors", "reward_model.safetensors"]
    if result.get("lora_names") == expected_loras:
        results.record_pass("Multiple flags combined")
    else:
        results.record_fail("Multiple flags", f"Expected {expected_loras}, got {result.get('lora_names')}")
    
    results.summary()
    assert results.failed == 0, f"{results.failed} sub-test(s) failed"


def test_combined_formats():
    """Test combinations of multiple formats"""
    print("\n" + "="*70)
    print("TEST: Combined Formats")
    print("="*70)
    
    results = TestResult()
    
    # Test: activated_loras + additional_loras + feature flags
    print("\nTest: Multiple format sources combined")
    params = {
        "activated_loras": ["explicit.safetensors"],
        "loras_multipliers": "1.2",
        "additional_loras": {
            "https://example.com/downloaded.safetensors": 0.9
        },
        "use_causvid_lora": True
    }
    
    # Process through pipeline
    result = process_format_1_wgp_legacy(params)
    result = process_format_5_additional_dict(result)
    result = process_format_8_feature_flags(result, TEST_MODEL_TYPE)
    
    expected_loras = ["explicit.safetensors", "downloaded.safetensors", "CausVid"]
    if set(result.get("lora_names", [])) == set(expected_loras):
        results.record_pass("Combined formats - all LoRAs present")
    else:
        results.record_fail("Combined formats", f"Expected {expected_loras}, got {result.get('lora_names')}")
    
    # Check multipliers match
    if len(result.get("lora_multipliers", [])) == len(result.get("lora_names", [])):
        results.record_pass("Combined formats - multipliers match names")
    else:
        results.record_fail("Multipliers length", f"Names: {len(result.get('lora_names', []))}, Mults: {len(result.get('lora_multipliers', []))}")
    
    results.summary()
    assert results.failed == 0, f"{results.failed} sub-test(s) failed"


def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n" + "="*70)
    print("TEST: Edge Cases")
    print("="*70)
    
    results = TestResult()
    
    # Test: Empty lists
    print("\nTest: Empty lists")
    params = {
        "activated_loras": [],
        "loras_multipliers": ""
    }
    result = process_format_1_wgp_legacy(params)
    
    if result.get("lora_names", []) == []:
        results.record_pass("Empty lists handled")
    else:
        results.record_fail("Empty lists", f"Got {result.get('lora_names')}")
    
    # Test: Missing multipliers
    print("\nTest: Missing multipliers")
    params = {
        "lora_names": ["lora1.safetensors", "lora2.safetensors"]
        # No multipliers provided
    }
    result = process_format_3_internal_list(params)
    
    if "lora_names" in result:
        results.record_pass("Missing multipliers doesn't crash")
    else:
        results.record_fail("Missing multipliers", "Unexpected error")
    
    # Test: Whitespace handling
    print("\nTest: Whitespace handling")
    params = {
        "activated_loras": " lora1.safetensors , lora2.safetensors ",
        "loras_multipliers": "1.0, 0.8"  # Comma-separated (matching the loras format)
    }
    result = process_format_1_wgp_legacy(params)
    
    if result.get("lora_names") == ["lora1.safetensors", "lora2.safetensors"]:
        results.record_pass("Whitespace trimmed")
    else:
        results.record_fail("Whitespace", f"Got {result.get('lora_names')}")
    
    # Test: Deduplication
    print("\nTest: Deduplication")
    params = {
        "lora_names": ["lora1.safetensors"],
        "lora_multipliers": [1.0],
        "additional_loras": {
            "lora1.safetensors": 0.8  # Duplicate
        }
    }
    result = process_format_5_additional_dict(params)
    
    # Current system doesn't deduplicate - just checking it doesn't crash
    if "lora_names" in result:
        results.record_pass("Duplicate handling doesn't crash")
    else:
        results.record_fail("Deduplication", "Unexpected error")
    
    results.summary()
    assert results.failed == 0, f"{results.failed} sub-test(s) failed"


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all baseline tests."""
    print("\n" + "="*70)
    print("LORA FORMAT BASELINE TEST SUITE")
    print("Testing current system behavior before refactoring")
    print("="*70)
    
    all_passed = True
    
    # Run all test suites
    all_passed &= test_format_1_wgp_legacy()
    all_passed &= test_format_3_internal_list()
    all_passed &= test_format_5_additional_dict()
    all_passed &= test_format_6_qwen_task()
    all_passed &= test_format_8_feature_flags()
    all_passed &= test_combined_formats()
    all_passed &= test_edge_cases()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Baseline established")
        print("\nYou can now refactor with confidence!")
        print("Run this test again after refactoring to validate behavior is preserved.")
    else:
        print("❌ SOME TESTS FAILED - Review failures above")
        print("\nNote: Some failures may be expected (e.g., deduplication not implemented)")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

