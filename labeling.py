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
            # 2-field format: uttID label (2021 eval protocol)
            elif len(parts) == 2:
                # 2-field format: uttID label (2021 eval protocol)
                utt, lbl = parts
                # files are .flac under data_root or data_root/flac
                fname = f"{utt}.flac"
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
    import argparse
    parser = argparse.ArgumentParser(description="Generate audio metadata")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Raw external data mode
    raw_parser = subparsers.add_parser("raw", help="Generate metadata for raw fake/real folders")
    raw_parser.add_argument("data_dir", help="Root folder with fake/ and real/ subdirs")
    raw_parser.add_argument("output_file", help="Path to write metadata (e.g., external_label.txt)")
    raw_parser.add_argument("--speaker_prefix", default="EXT", help="Speaker ID prefix")

    # ASVspoof CM mode
    cm_parser = subparsers.add_parser("cm", help="Generate metadata from ASVspoof CM protocol")
    cm_parser.add_argument("protocol_path", help="Path to ASVspoof CM trial metadata file")
    cm_parser.add_argument("data_root", help="Root folder containing audio files (may include flac/ subdir)")
    cm_parser.add_argument("output_file", help="Path to write metadata (e.g., cm_metadata.txt)")
    cm_parser.add_argument("--subsets", nargs="+", default=["progress"], help="Subset(s) to include (e.g., progress dev eval)")
    cm_parser.add_argument("--speaker_prefix", default="LA", help="Speaker ID prefix for ASVspoof data")

    args = parser.parse_args()

    if args.mode == "raw":
        generate_metadata(args.data_dir, args.output_file, args.speaker_prefix)
    elif args.mode == "cm":
        generate_asvspoof_cm_metadata(
            protocol_path=args.protocol_path,
            data_root=args.data_root,
            subsets=tuple(args.subsets),
            output_file=args.output_file,
            speaker_prefix=args.speaker_prefix
        )