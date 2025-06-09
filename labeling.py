"""
labeling.py

Generate a metadata file for audio datasets in the format:
  SPEAKER FILENAME - - LABEL

Assumes directory structure:
  external_data/
    fake/
      *.wav or *.mp3
    real/
      *.wav or *.mp3
"""

import os
import argparse

def generate_metadata(data_dir, output_file, speaker_prefix='EXT'):
    """
    Walk through 'fake' and 'real' subdirs of data_dir and write lines:
      SPEAKER FILENAME - - LABEL
    """
    labels = {'fake': 'Fake', 'real': 'Real'}
    with open(output_file, 'w') as f:
        for subdir, label_str in labels.items():
            dir_path = os.path.join(data_dir, subdir)
            if not os.path.isdir(dir_path):
                continue
            for fname in sorted(os.listdir(dir_path)):
                if not fname.lower().endswith(('.wav')):
                    continue
                # include the subfolder in the relative file path
                rel_path = os.path.join(subdir, fname)
                f.write(f"{speaker_prefix} {rel_path} - - {label_str}\n")


# ASVspoof LA CM metadata generation
def generate_asvspoof_cm_metadata(protocol_path, data_root, subsets=('progress',), output_file='cm_metadata.txt', speaker_prefix='LA'):
    """
    Generate CM metadata for ASVspoof LA track.
    Only include trials whose subset is in `subsets` (e.g., 'progress').
    Writes lines in format: SPEAKER REL_PATH - - LABEL
    """
    import os
    label_map = {'spoof': 'Fake', 'bona-fide': 'Real', 'bonafide': 'Real'}
    with open(protocol_path, 'r') as proto, open(output_file, 'w') as out:
        for line in proto:
            parts = line.strip().split()
            # 8-field format: spk utt codec lang attack lbl notrim subset
            if len(parts) == 8:
                spk, utt, codec, lang, attack, lbl, _, subset = parts
                if subset not in subsets:
                    continue
                fname = f"{utt}-{codec}-{lang}.wav"
                rel_path = os.path.join(subset, fname)
            # 5-field format: spk utt - - lbl (train/dev protocol)
            elif len(parts) == 5:
                spk, utt, dash1, dash2, lbl = parts
                # train/dev protocols list files directly under data_root (no subfolder)
                fname = f"{utt}.wav"
                rel_path = fname
            else:
                continue  # unexpected format
            # Search for file under data_root or data_root/flac with .wav or .flac
            base_name, _ = os.path.splitext(rel_path)
            found_path = None
            for sub in ["", "flac"]:
                for ext in [".wav", ".flac"]:
                    candidate = os.path.join(data_root, sub, base_name + ext)
                    if os.path.exists(candidate):
                        found_path = candidate
                        # Update rel_path relative to data_root
                        rel_path = os.path.relpath(candidate, data_root)
                        break
                if found_path:
                    break
            if not found_path:
                print(f"[WARN] Missing file: {os.path.join(data_root, rel_path)}")
                continue
            # write metadata line
            out.write(f"{speaker_prefix} {rel_path} - - {label_map.get(lbl, lbl)}\n")

if __name__ == "__main__":
    """
    python labeling.py {datapath} {outputpath}
    """
    p = argparse.ArgumentParser(description="Generate audio metadata")
    p.add_argument("data_dir", help="Root folder with fake/ and real/")
    p.add_argument("output_file", help="Where to write metadata (e.g. external_label.txt)")
    p.add_argument("--speaker_prefix", default="EXT", help="Speaker ID prefix")
    args = p.parse_args()
    generate_metadata(args.data_dir, args.output_file, args.speaker_prefix)