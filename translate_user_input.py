#!/usr/bin/env python3
"""
Translate user_input column in CSV from English to Bahasa Indonesia.

This standalone script translates questions while preserving technical terms
and regulatory names used in Indonesian government documents.
"""

import os
import csv
import json
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import yaml

# Load environment variables
load_dotenv()


def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Translate user_input from English to Indonesian",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file (default: multihop_partial.csv)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file (default: multihop_translated.csv)"
    )

    return parser.parse_args()


# Parse arguments and load configuration
args = parse_args()

try:
    config = load_config(args.config)
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit(1)

# Configuration
INPUT_FILE = args.input if args.input else "multihop_partial.csv"
OUTPUT_FILE = args.output if args.output else "multihop_translated.csv"
PROGRESS_FILE = "translation_progress.json"

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Translation settings from config
TRANSLATION_MODEL = config['translation']['model']
TRANSLATION_TEMPERATURE = config['translation']['temperature']
TRANSLATION_MAX_TOKENS = config['translation']['max_tokens']


def load_progress():
    """Load translation progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"last_row": 0, "translations": {}}


def save_progress(progress):
    """Save translation progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def translate_to_indonesian(text, row_num):
    """
    Translate English text to Bahasa Indonesia using OpenRouter.
    Preserves technical terms and regulatory names.

    Args:
        text: English text to translate
        row_num: Row number for logging

    Returns:
        Translated Indonesian text
    """
    system_prompt = """Anda adalah penerjemah profesional. Tugas Anda adalah MENERJEMAHKAN pertanyaan dari bahasa Inggris ke Bahasa Indonesia.

PENTING - JANGAN JAWAB PERTANYAAN. HANYA TERJEMAHKAN.

Terjemahkan PERTANYAAN bahasa Inggris berikut ke dalam PERTANYAAN Bahasa Indonesia yang setara.

ATURAN:
1. PERTAHANKAN semua istilah teknis dan nama resmi dalam bentuk aslinya:
   - Peraturan Pemerintah, Peraturan Daerah, Undang-Undang, Peraturan Menteri
   - APBD, RPJMD, RKPD, DPRD, Bappeda
   - Gubernur, Kepala Daerah, Menteri Dalam Negeri
   - Nomor dan tahun peraturan (contoh: "Nomor 12 Tahun 2019")

2. TERJEMAHKAN struktur pertanyaan (What/How/Why ke Apa/Bagaimana/Mengapa)

3. PERTAHANKAN format pertanyaan - jika input adalah pertanyaan, output HARUS pertanyaan juga

4. Output HANYA teks terjemahan pertanyaan, TANPA jawaban atau penjelasan

Contoh BENAR:
Input: "What does Peraturan Pemerintah Nomor 12 Tahun 2019 say about regional financial management?"
Output: "Apa yang diatur dalam Peraturan Pemerintah Nomor 12 Tahun 2019 tentang pengelolaan keuangan daerah?"

Input: "How does Undang-Undang Dasar 1945 serve as a basis for regional regulations?"
Output: "Bagaimana Undang-Undang Dasar 1945 menjadi dasar bagi peraturan daerah?"

Contoh SALAH (JANGAN LAKUKAN INI):
Input: "What does Peraturan Pemerintah say?"
Output SALAH: "Peraturan Pemerintah mengatur tentang..." ← INI JAWABAN, BUKAN TERJEMAHAN!
Output BENAR: "Apa yang diatur Peraturan Pemerintah?" ← INI TERJEMAHAN PERTANYAAN"""

    try:
        response = client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=TRANSLATION_TEMPERATURE,
            max_tokens=TRANSLATION_MAX_TOKENS
        )

        translated = response.choices[0].message.content.strip()
        print(f"Row {row_num}: Translated")
        return translated

    except Exception as e:
        print(f"Row {row_num}: Error - {str(e)}")
        return text  # Return original text on error


def main():
    """Main translation function."""

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available CSV files: {list(Path('.').glob('*.csv'))}")
        return

    # Load progress
    progress = load_progress()
    last_row = progress.get("last_row", 0)
    translations = progress.get("translations", {})

    print("=" * 60)
    print("Translation: English to Bahasa Indonesia")
    print("=" * 60)
    print(f"Input:       {INPUT_FILE}")
    print(f"Output:      {OUTPUT_FILE}")
    print(f"Model:       {TRANSLATION_MODEL}")
    print(f"Temperature: {TRANSLATION_TEMPERATURE}")
    print(f"Max Tokens:  {TRANSLATION_MAX_TOKENS}")
    print(f"Resume:      Starting from row {last_row + 1}")
    print("=" * 60)
    print()

    # Read CSV and translate
    rows = []
    total_rows = 0
    translated_count = 0
    cached_count = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for idx, row in enumerate(reader, start=1):
            total_rows += 1

            # Check if already translated
            if str(idx) in translations:
                row['user_input'] = translations[str(idx)]
                cached_count += 1
                print(f"Row {idx}: Using cached translation")
            elif idx > last_row:
                # Translate
                original = row['user_input']
                translated = translate_to_indonesian(original, idx)
                row['user_input'] = translated
                translated_count += 1

                # Save progress
                translations[str(idx)] = translated
                progress['last_row'] = idx
                progress['translations'] = translations
                save_progress(progress)
            else:
                cached_count += 1
                print(f"Row {idx}: Skipped (already processed)")

            rows.append(row)

    # Write output CSV
    print()
    print("=" * 60)
    print(f"Writing translated CSV to {OUTPUT_FILE}")
    print("=" * 60)

    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Translation complete!")
    print(f"  Total rows:       {total_rows}")
    print(f"  Newly translated: {translated_count}")
    print(f"  From cache:       {cached_count}")
    print()
    print("Next steps:")
    print(f"1. Review {OUTPUT_FILE} to verify translations")
    print(f"2. If satisfied, use it for your RAG evaluation")
    print(f"3. To clean up, delete {PROGRESS_FILE}")
    print()


if __name__ == "__main__":
    main()
