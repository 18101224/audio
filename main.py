import torchaudio
import torch
from mir_eval import separation
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.utils import download_asset
from torchaudio.transforms import Fade
import numpy as np
import time
def separate_sources(
    model,
    mix,
    segment=10.0,
    overlap=0.1,
    device=None,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

def resample(input_data, sample_rate, target_sample_rate):
    ratio = float(target_sample_rate) / sample_rate
    output_data = np.interp(np.arange(len(input_data)) * ratio, np.arange(len(input_data)), input_data)
    return output_data

if __name__ == '__main__' :
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    print(model.audio_channels)
    device = torch.device('cuda')

    model=model.to(device)

    sample_rate = bundle.sample_rate

    print(f"Sample rate: {sample_rate}")

    start = time.time()
    waveform, sr = torchaudio.load('../data/audio/genres_original/rock/rock.00036.wav')
    waveform = torchaudio.functional.resample(waveform,sr,sample_rate)
    waveform = waveform.to(device)
    waveform = torch.concat([waveform]*2,dim=0)
    mixture = waveform

    segment : int=10
    overlap = 0.1
    print('Separating track')
    ref = waveform.mean(0)
    waveform = (waveform-ref.mean())/ref.std()

    sources = separate_sources(
        model,waveform[None],segment=segment,overlap=overlap
    )[0]
    sources = sources * ref.std()+ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list,sources))

    for name, wave in audios.items():
        torchaudio.save(f'{name}.wav',wave.detach().cpu(), format='wav',sample_rate=sample_rate)
    end = time.time()
    print(f'time :{end-start}')