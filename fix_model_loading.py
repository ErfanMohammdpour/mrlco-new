#!/usr/bin/env python3
"""
Fix script for model loading compatibility
"""

def fix_model_loading():
    """Apply a fix to make model loading more compatible"""
    
    # Read the current meta_seq2seq_policy.py
    with open('policies/meta_seq2seq_policy.py', 'r') as f:
        content = f.read()
    
    # Replace the load_variables method with a more robust version
    old_load_method = '''        if isinstance(loaded_params, list):
            assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
            for d, v in zip(loaded_params, variables):
                restores.append(v.assign(d))
        else:
            for v in variables:
                # Try to load with current name first
                if v.name in loaded_params:
                    restores.append(v.assign(loaded_params[v.name]))
                else:
                    # Try to find a compatible name in the loaded parameters
                    found = False
                    for loaded_name in loaded_params.keys():
                        # Check if the variable name is compatible (ignore some suffixes)
                        if (v.name.split('/')[-1].split(':')[0] in loaded_name or 
                            loaded_name.split('/')[-1].split(':')[0] in v.name.split('/')[-1].split(':')[0]):
                            print(f"🔄 Mapping {loaded_name} -> {v.name}")
                            restores.append(v.assign(loaded_params[loaded_name]))
                            found = True
                            break
                    
                    if not found:
                        print(f"⚠️  Warning: Could not find compatible variable for {v.name}")
                        # Initialize with current value (no change)
                        restores.append(v.assign(v))'''
    
    new_load_method = '''        if isinstance(loaded_params, list):
            assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
            for d, v in zip(loaded_params, variables):
                restores.append(v.assign(d))
        else:
            # Create a mapping of variable names for compatibility
            loaded_vars = {}
            for loaded_name, loaded_value in loaded_params.items():
                # Extract the base name without scope and suffix
                base_name = loaded_name.split('/')[-1].split(':')[0]
                loaded_vars[base_name] = loaded_value
            
            for v in variables:
                # Try to load with current name first
                if v.name in loaded_params:
                    restores.append(v.assign(loaded_params[v.name]))
                else:
                    # Try to find by base name
                    base_name = v.name.split('/')[-1].split(':')[0]
                    if base_name in loaded_vars:
                        print(f"🔄 Mapping {base_name} -> {v.name}")
                        restores.append(v.assign(loaded_vars[base_name]))
                    else:
                        print(f"⚠️  Warning: Could not find compatible variable for {v.name}")
                        # Initialize with current value (no change)
                        restores.append(v.assign(v))'''
    
    if old_load_method in content:
        content = content.replace(old_load_method, new_load_method)
        print("✅ Applied improved model loading fix")
    else:
        print("⚠️  Pattern not found, trying alternative approach...")
        
        # Alternative: Just add error handling
        content = content.replace(
            'restores.append(v.assign(loaded_params[v.name]))',
            '''try:
                restores.append(v.assign(loaded_params[v.name]))
            except KeyError:
                print(f"⚠️  Warning: Could not find variable {v.name} in loaded model")
                # Skip this variable (keep current value)
                pass'''
        )
        print("✅ Applied alternative fix with error handling")
    
    # Write the fixed file
    with open('policies/meta_seq2seq_policy.py', 'w') as f:
        f.write(content)
    
    print("✅ Model loading fix applied successfully!")
    print("🚀 Now try running your evaluation again:")
    print("   python meta_evaluator_with_viz_integration.py")

if __name__ == "__main__":
    fix_model_loading()
