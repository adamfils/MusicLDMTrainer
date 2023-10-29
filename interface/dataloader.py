# Filename: data_loader.py

import json
import os

import torch
import torchaudio
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader


class MusicDataset(Dataset):
    def __init__(self, dataset_folders):
        self.dataset_folders = dataset_folders if isinstance(dataset_folders, list) else [dataset_folders]
        self.file_pairs = self._get_file_pairs()
        self.valid_indices = self._get_valid_indices()

    def _get_file_pairs(self):
        file_pairs = []
        for dataset_folder in self.dataset_folders:
            files = os.listdir(dataset_folder)
            audio_files = [f for f in files if f.endswith('.mp3') or f.endswith('.wav')]
            file_pairs.extend([(os.path.join(dataset_folder, af),
                                os.path.join(dataset_folder, af.replace('.mp3', '.json')))
                               for af in audio_files])
        return file_pairs

    def _get_valid_indices(self):
        valid_indices = []
        for idx, (audio_file, json_file) in enumerate(self.file_pairs):
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        json.load(f)  # Try to load the JSON to see if it causes a UnicodeDecodeError
                    valid_indices.append(idx)
                except:
                    # print(f"Failed to read {json_file}. Skipping.")
                    pass
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        true_idx = self.valid_indices[idx]
        audio_file, json_file = self.file_pairs[true_idx]

        # convert mp3 to wav if necessary
        if audio_file.endswith(".mp3"):
            wav_audio_path = audio_file.replace(".mp3", ".wav")
            if not os.path.exists(wav_audio_path):  # convert if .wav doesn't exist
                sound = AudioSegment.from_mp3(audio_file)
                sound.export(wav_audio_path, format="wav")
            audio_file = wav_audio_path

        audio, _ = torchaudio.load(audio_file)
        # audio = audio[:, :self.fixed_length]  # Add this line to fix the issue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            # print(f"Failed to read {json_file}. Skipping.")
            return None  # This will not be reached if valid_indices is properly set up

        prompt = data['description']

        return audio, prompt


def trim_sequences(batch):
    # Find the length of the shortest audio in the batch
    min_length = min(audio.size(1) for audio, _ in batch)
    # Trim longer audios to the length of the shortest audio
    trimmed_audio = [audio[:, :min_length] for audio, _ in batch]
    # Get the text data
    text = [text for _, text in batch]
    return torch.stack(trimmed_audio), text


def get_dataloader(dataset_folder, batch_size: int = 50, shuffle: bool = True):
    dataset = MusicDataset(dataset_folder)

    def collate_fn(batch):
        audio, text = zip(*batch)
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)
        return audio, text  # Adjust this line to pad the sequences

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=trim_sequences)
    return dataloader
