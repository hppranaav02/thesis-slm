import shutil
import os

# ...existing code...

def copy_tokenizer_model(tokenizer, output_dir, model_name):
    # Try to find tokenizer.model in the original model dir
    try:
        # Download/cached dir for the original model
        orig_dir = tokenizer.pretrained_vocab_files_map.get('tokenizer_file', {}).get(model_name)
        if orig_dir is None:
            orig_dir = tokenizer.pretrained_vocab_files_map.get('tokenizer_file', {}).get('default')
        if orig_dir is None:
            print("Could not find original tokenizer.model path.")
            return
        # Get the local path
        orig_path = tokenizer.convert_file_path(orig_dir)
        dest_path = os.path.join(output_dir, 'tokenizer.model')
        if os.path.exists(orig_path):
            shutil.copyfile(orig_path, dest_path)
            print(f"Copied tokenizer.model to {dest_path}")
        else:
            print(f"tokenizer.model not found at {orig_path}")
    except Exception as e:
        print(f"Error copying tokenizer.model: {e}")

# ...existing code...

# After tokenizer.save_pretrained(OUTPUT_DIR)
copy_tokenizer_model(tokenizer, OUTPUT_DIR, MODEL_NAME)
print("Checked and copied tokenizer.model if needed.")
# ...existing code...
