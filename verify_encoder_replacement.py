"""
Script to verify that all encoder references have been properly replaced.
Scans the codebase to ensure no references to old encoder remain.
"""
import os
import re


def scan_file_for_encoder_refs(filepath):
    """Scan a file for encoder references."""
    encoder_patterns = [
        r'create_encoder\s*\(',
        r'create_bidrect_encoder\s*\(',
        r'_build_encoder_cell\s*\(',
        r'self\.encoder_outputs.*=.*dynamic_rnn',
    ]
    
    found_refs = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for line_num, line in enumerate(content.split('\n'), 1):
            for pattern in encoder_patterns:
                if re.search(pattern, line):
                    # Check if it's in a comment
                    if not line.strip().startswith('#'):
                        # Check if it's the Graph2Seq import/usage
                        if 'graph2seq_encoder' not in line.lower():
                            found_refs.append({
                                'file': filepath,
                                'line': line_num,
                                'content': line.strip(),
                                'pattern': pattern
                            })
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return found_refs


def verify_graph2seq_imports(filepath):
    """Verify that files using encoder have Graph2Seq imports."""
    needs_import = False
    has_import = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if file uses encoder
        if 'encoder_outputs' in content or 'encoder_state' in content:
            needs_import = True
            
        # Check if has Graph2Seq import
        if 'from policies.graph2seq_encoder import' in content or 'import policies.graph2seq_encoder' in content:
            has_import = True
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        
    return needs_import, has_import


def scan_directory(root_dir):
    """Scan directory for encoder references."""
    all_refs = []
    missing_imports = []
    
    excluded_files = [
        'verify_encoder_replacement.py',
        'test_encoder_compatibility.py',
        'graph2seq_encoder.py'
    ]
    
    for root, dirs, files in os.walk(root_dir):
        # Skip Graph2Seq directory
        if 'Graph2Seq' in root:
            continue
            
        for file in files:
            if file.endswith('.py') and file not in excluded_files:
                filepath = os.path.join(root, file)
                
                # Check for old encoder references
                refs = scan_file_for_encoder_refs(filepath)
                all_refs.extend(refs)
                
                # Check for missing imports
                needs, has = verify_graph2seq_imports(filepath)
                if needs and not has and 'meta_seq2seq_policy' in file:
                    missing_imports.append(filepath)
    
    return all_refs, missing_imports


def main():
    """Main verification function."""
    print("Verifying encoder replacement in metarl-offloading project...")
    print("="*60)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Scan for old encoder references
    old_refs, missing_imports = scan_directory(project_root)
    
    if old_refs:
        print("\n[WARNING] Found references to old encoder implementation:")
        print("-"*60)
        for ref in old_refs:
            print(f"\nFile: {ref['file']}")
            print(f"Line {ref['line']}: {ref['content']}")
            print(f"Pattern matched: {ref['pattern']}")
    else:
        print("\n[OK] No references to old encoder implementation found!")
    
    if missing_imports:
        print("\n[WARNING] Files that may need Graph2Seq imports:")
        print("-"*60)
        for file in missing_imports:
            print(f"  - {file}")
    else:
        print("\n[OK] All necessary imports are in place!")
    
    # Check that Graph2Seq encoder is being used
    policy_file = os.path.join(project_root, 'policies', 'meta_seq2seq_policy.py')
    
    if os.path.exists(policy_file):
        with open(policy_file, 'r') as f:
            content = f.read()
            
        if 'create_graph2seq_encoder' in content:
            print("\n[OK] Graph2Seq encoder is properly integrated in meta_seq2seq_policy.py")
        else:
            print("\n[WARNING] Graph2Seq encoder not found in meta_seq2seq_policy.py")
    
    # Summary
    print("\n" + "="*60)
    if not old_refs and not missing_imports:
        print("[OK] VERIFICATION PASSED: Encoder replacement is complete!")
    else:
        print("[WARNING] VERIFICATION FAILED: Some issues need to be addressed.")
    print("="*60)


if __name__ == "__main__":
    main()