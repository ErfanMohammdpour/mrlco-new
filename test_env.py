import sys
import os

print("Python path:")
print(sys.path)

print("\nTrying to import environment module...")

try:
    from env.mec_offloading_envs.offloading_env import Resources, OffloadingEnvironment
    print("✓ Environment module imported successfully")
except Exception as e:
    print(f"✗ Failed to import environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nCreating resource cluster...")
try:
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0, 
        bandwidth_dl=7.0
    )
    print("✓ Resource cluster created")
except Exception as e:
    print(f"✗ Failed to create resource cluster: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nCreating minimal environment (1 graph, batch_size=1)...")
try:
    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=1,  # Minimal
        graph_number=1,  # Minimal
        graph_file_paths=[
            "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_1/random.20.",
        ],
        time_major=False
    )
    print("✓ Environment created successfully!")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nEnvironment test completed successfully!")