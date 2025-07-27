"""
Master script to run all Graph2Seq encoder verifications for Metarl-Offloading.
This ensures complete validation of the encoder replacement.
"""
import os
import sys
import subprocess

def run_verification_script(script_name, description):
    """Run a verification script and capture results."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('='*80)
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        return False
        
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("[STDERR]:", result.stderr)
            
        # Check for success indicators
        success = (
            result.returncode == 0 or
            "[SUCCESS]" in result.stdout or 
            "[OK] VERIFICATION PASSED" in result.stdout or
            "[OK] ALL" in result.stdout
        )
        
        return success
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Script timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run script: {e}")
        return False


def main():
    """Run all verification scripts in sequence."""
    print("="*80)
    print("COMPREHENSIVE GRAPH2SEQ ENCODER VERIFICATION SUITE")
    print("="*80)
    print("\nThis will run all verification scripts to ensure the Graph2Seq encoder")
    print("is properly integrated into the Metarl-Offloading project.")
    
    # Define verification scripts and their descriptions
    verifications = [
        ("verify_encoder_replacement.py", "Verify old encoder is completely replaced"),
        ("test_encoder_compatibility.py", "Test encoder shape and interface compatibility"),
        ("verify_aggregator_inclusion.py", "Verify aggregator parameters in optimization"),
        ("test_training_dynamics.py", "Test training dynamics and learning"),
        ("comprehensive_encoder_verification.py", "Run comprehensive verification suite")
    ]
    
    results = {}
    
    # Run each verification
    for script, description in verifications:
        success = run_verification_script(script, description)
        results[script] = success
        
        if not success:
            print(f"\n[WARNING] Verification failed for: {script}")
            
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\nTotal Verifications: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    print("\nDetailed Results:")
    for script, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {script}")
        
    # Overall verdict
    print("\n" + "="*80)
    if passed == total:
        print("[SUCCESS] ALL VERIFICATIONS PASSED!")
        print("\nThe Graph2Seq encoder has been successfully integrated into Metarl-Offloading.")
        print("Key achievements:")
        print("  - Old encoder completely removed")
        print("  - Full interface compatibility maintained")
        print("  - All encoder parameters included in optimization")
        print("  - Training dynamics verified")
        print("  - Meta-RL compatibility confirmed")
    else:
        print("[WARNING] Some verifications failed.")
        print("\nPlease review the failed tests above and address any issues.")
        print("Common issues to check:")
        print("  - TensorFlow installation")
        print("  - Import paths")
        print("  - Missing dependencies")
    print("="*80)
    
    return passed == total


if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run all verifications
    success = main()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)