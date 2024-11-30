import pandas as pd
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import numpy as np
import noisereduce as nr
import soundfile as sf

def load_process(path):
    df = pd.read_csv(f'{path}/dem-info.csv')
    # fix 66* string value in Age column
    df['Age'] = df['Age'].apply(lambda x: x.replace('66*', '66'))
    df['Age'] = df['Age'].astype(int)
    # get_file_names
    for ext in ["CTD", "PFT", 'SFT']:
        df[f'{ext}_wav'] = f'{path}/' + df['Record-ID'] + '/' + df['Record-ID'] + f'__{ext}.wav'
        df[f'{ext}_txt'] = f'{path}/' + df['Record-ID'] + '/' + df['Record-ID'] + f'__{ext}.txt'

    for ext in ["CTD", "PFT", 'SFT']:
        df[f'{ext}_fulltext'] = df[f'{ext}_txt'].apply(lambda x: Path(x).read_text())

    return df

process_path = './PROCESS-V1'
df = load_process(process_path)

df_wav = df[['CTD_wav', 'PFT_wav', 'SFT_wav']].copy()
df_audio_ctd = pd.DataFrame()
df_audio_pft = pd.DataFrame()
df_audio_sft = pd.DataFrame()

def read_wav(df):
    y = []
    sr = []
    for wav in df:
        y, sr = librosa.load(wav, sr=None)
        y.append(y)
        sr.append(sr)
    df_audio_sr = pd.DataFrame()
    df_audio_sr['Audio'] = y
    df_audio_sr['Sampling Rate'] = sr
    return df_audio_sr

df_audio_ctd = read_wav(df_wav['CTD_wav'])
df_audio_pft = read_wav(df_wav['PFT_wav'])
df_audio_sft = read_wav(df_wav['SFT_wav'])

def display_first_five_audio(df):
    for _, row in df.head(5).iterrows():
        y = row['Audio']
        sr = row['Sampling Rate']
        display(Audio(y, rate=sr))

def spect_first_five_audio(df):
    for _, row in df.head(5).iterrows():
        y = row['Audio']
        sr = row['Sampling Rate']

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.xlabel('Zeit (s)')
        plt.ylabel('Frequenz (Hz)')
        plt.show()

def denoise_audio(df):
    for index, row in df.iterrows():
        y = row['Audio']
        sr = row['Sampling Rate']
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        df.at[index, 'Audio'] = y_denoised
    return df


def normalize_audio(df, target_rms=0.1):
    for index, row in df.iterrows():
        y = row['Audio']
        rms = np.sqrt(np.mean(y**2))  # RMS-Wert berechnen
        scaling_factor = target_rms / rms
        df.at[index, 'Audio'] = y * scaling_factor  # Skalieren auf Ziel-RMS
    return df

df_ctd_denoised = denoise_audio(df_audio_ctd.copy())
df_ctd_clean = normalize_audio(df_ctd_denoised.copy())
df_pft_denoised = denoise_audio(df_audio_pft.copy())
df_pft_clean = normalize_audio(df_pft_denoised.copy())
df_sft_denoised = denoise_audio(df_audio_sft.copy())
df_sft_clean = normalize_audio(df_sft_denoised.copy())

df_ctd_clean["Record-ID"] = df["Record-ID"]
df_pft_clean["Record-ID"] = df["Record-ID"]
df_sft_clean["Record-ID"] = df["Record-ID"]

def save_process(df, path, ext):
    for index, row in df.iterrows():
        y_clean = row['Audio']
        sr = row['Sampling Rate']
        id = row['Record-ID']

        output_filename = f"{path}/{id}/{id}_{ext}_clean.wav"
        sf.write(output_filename, y_clean, sr)

#save_process(df_ctd_clean, process_path, 'CTD')
#save_process(df_pft_clean, process_path, 'PFT')
#save_process(df_sft_clean, process_path, 'SFT')