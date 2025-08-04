import sys
print("Python path:", sys.path)

print("Testing imports...")
try:
    from env.mec_offloaing_envs.offloading_env import Resources
    print("✓ Resources imported")
except Exception as e:
    print("✗ Failed to import Resources:", e)

try:
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    print("✓ OffloadingEnvironment imported")
except Exception as e:
    print("✗ Failed to import OffloadingEnvironment:", e)

print("\nCreating resource cluster...")
try:
    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)
    print("✓ Resource cluster created")
except Exception as e:
    print("✗ Failed to create resource cluster:", e)

print("\nTesting with smaller graph set...")
try:
    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=10,  # Smaller batch
                                graph_number=10,  # Fewer graphs
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                ],
                                time_major=False)
    print("✓ Environment created successfully!")
except Exception as e:
    print("✗ Failed to create environment:", e)
    import traceback
    traceback.print_exc()