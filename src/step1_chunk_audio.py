import os
import glob
import librosa
import numpy as np
import soundfile as sf

def chunk_audio_pipeline(
    raw_base_dir="data/01_raw",
    chunk_base_dir="data/02_chunks",
    chunk_duration=5,
    sr=16000,
    top_db=30,
    min_duration=2.5
):
    """
    Memotong setiap file audio dari data/01_raw menjadi potongan 5 detik (80.000 sampel pada 16.000 Hz)
    dengan aturan:
    1. Safe silence trimming (top_db=30). Jika hasil trim < 2.5s, gunakan sinyal asli.
    2. Potong per 5 detik tanpa overlap.
    3. Jika sisa potongan >= 2.5s, lakukan zero-padding ke 5 detik penuh. Jika < 2.5s, abaikan.
    4. Format nama: {parent_id}__chunk{idx:02d}.wav untuk mencegah data leakage.
    """
    classes = ["ramai", "tidak_ramai"]
    samples_per_chunk = int(chunk_duration * sr)
    min_samples = int(min_duration * sr)

    stats = {
        "raw_counts": {},
        "chunk_counts": {},
        "processed_files": {},
        "failed_files": []
    }

    valid_exts = ('.wav', '.mp3', '.m4a', '.ogg', '.flac', '.mpeg')

    for cls in classes:
        input_folder = os.path.join(raw_base_dir, cls)
        output_folder = os.path.join(chunk_base_dir, cls)
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_folder):
            print(f"[PERINGATAN] Folder input tidak ditemukan: {input_folder}")
            continue

        raw_files = [f for f in sorted(os.listdir(input_folder)) if f.lower().endswith(valid_exts)]
        stats["raw_counts"][cls] = len(raw_files)
        stats["chunk_counts"][cls] = 0
        stats["processed_files"][cls] = 0

        print(f"\n==================================================")
        print(f" Memproses Kelas: {cls.upper()} ({len(raw_files)} file)")
        print(f"==================================================")

        for fname in raw_files:
            filepath = os.path.join(input_folder, fname)
            parent_id = os.path.splitext(fname)[0]

            try:
                # 1. Load audio mono pada 16.000 Hz
                y, file_sr = librosa.load(filepath, sr=sr, mono=True)

                if y.size == 0:
                    stats["failed_files"].append({
                        "file": fname,
                        "class": cls,
                        "reason": "Audio kosong (0 sampel)"
                    })
                    print(f"  [GAGAL] {fname} -> Audio kosong")
                    continue

                # 2. Safe silence trimming (top_db=30)
                y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
                if len(y_trimmed) < min_samples:
                    # Durasi trim < 2.5s, gunakan sinyal asli agar kelas tidak_ramai tidak terhapus
                    y_proc = y
                    trim_status = "Batal Trim (Sinyal Asli)"
                else:
                    y_proc = y_trimmed
                    trim_status = "Trim Berhasil"

                total_len = len(y_proc)

                # Jika total durasi bahkan kurang dari 2.5s, skip
                if total_len < min_samples:
                    stats["failed_files"].append({
                        "file": fname,
                        "class": cls,
                        "reason": f"Durasi total terlalu pendek (< {min_duration}s): {total_len/sr:.2f}s"
                    })
                    print(f"  [SKIP]  {fname} -> Terlalu pendek ({total_len/sr:.2f}s)")
                    continue

                # 3. Potong menjadi segmen 5 detik (80.000 sampel)
                full_chunks = total_len // samples_per_chunk
                remainder = total_len % samples_per_chunk
                chunk_list = []

                # Potongan penuh
                for i in range(full_chunks):
                    start = i * samples_per_chunk
                    end = start + samples_per_chunk
                    chunk_list.append(y_proc[start:end])

                # 4. Evaluasi sisa potongan
                if remainder >= min_samples:
                    start = full_chunks * samples_per_chunk
                    rem_chunk = y_proc[start:]
                    pad_len = samples_per_chunk - len(rem_chunk)
                    rem_chunk_padded = np.pad(rem_chunk, (0, pad_len), mode="constant")
                    chunk_list.append(rem_chunk_padded)

                # Simpan seluruh potongan
                for chunk_idx, chunk_data in enumerate(chunk_list, 1):
                    chunk_filename = f"{parent_id}__chunk{chunk_idx:02d}.wav"
                    chunk_path = os.path.join(output_folder, chunk_filename)
                    sf.write(chunk_path, chunk_data, sr)
                    stats["chunk_counts"][cls] += 1

                stats["processed_files"][cls] += 1
                print(f"  [OK]    {fname:20s} -> {len(chunk_list):2d} chunk(s) | {trim_status} | Durasi: {total_len/sr:.1f}s")

            except Exception as e:
                err_msg = str(e)
                stats["failed_files"].append({
                    "file": fname,
                    "class": cls,
                    "reason": f"{type(e).__name__}: {err_msg}"
                })
                print(f"  [GAGAL] {fname} -> {type(e).__name__}: {err_msg}")

    print("\n" + "=" * 55)
    print(" RINGKASAN EKSEKUSI TAHAP 3 (AUDIO CHUNKING)")
    print("=" * 55)
    for cls in classes:
        print(f" - Kelas {cls:12s} : {stats['processed_files'].get(cls, 0)}/{stats['raw_counts'].get(cls, 0)} file diproses -> {stats['chunk_counts'].get(cls, 0)} chunks (.wav)")
    print(f" - Total Chunk Dihasilkan : {sum(stats['chunk_counts'].values())} file .wav")
    print(f" - Total File Gagal/Lewat : {len(stats['failed_files'])} file")
    if stats["failed_files"]:
        print("\n Daftar File Gagal:")
        for fail in stats["failed_files"]:
            print(f"   * [{fail['class']}] {fail['file']}: {fail['reason']}")
    print("=" * 55)

    return stats

if __name__ == "__main__":
    chunk_audio_pipeline()
