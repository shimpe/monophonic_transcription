import crepe
from scipy.io import wavfile
import sys
import pathlib
import math
import plotly.graph_objects as go
from midiutil.MidiFile import MIDIFile
import statistics
import numpy as np

SEGMENT_LENGTH = 25
MIN_CONFIDENCE = 0.5
MAX_CONFIDENCE_REST = 0.25

COLOR_EDGES = ['#C232FF',
               '#C232FF',
               '#89FFAE',
               '#FFFF8B',
               '#A9E3FF',
               '#FF9797',
               '#A5FFE8',
               '#FDB0F8',
               '#FFDC9C',
               '#F3A3C4',
               '#E7E7E7']

COLOR = ['#D676FF',
         '#D676FF',
         '#0AFE57',
         '#FEFF00',
         '#56C8FF',
         '#FF4C4C',
         '#4CFFD1',
         '#FF4CF4',
         '#FFB225',
         '#C25581',
         '#737D73']


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier if n is not None else None


def cpsmidi(value):
    return math.log2(value * 0.0022727272727) * 12.0 + 69


def filtered(data, min_confidence, max_confidence):
    return [(d[0], round_half_up(d[1]), d[2], d[3]) if (min_confidence <= d[2] <= max_confidence) else (d[0], None, d[2], d[3]) for
            d in data]


def split_notes_rests(data):
    return filtered(data, MIN_CONFIDENCE, 1), filtered(data, 0, MAX_CONFIDENCE_REST)


def simplified(data):
    notes, rests = split_notes_rests(data)
    previous_note = None
    previous_volume = None
    previous_note_start_idx = 0
    midi_data = []
    for idx, (n, r) in enumerate(zip(notes, rests)):
        if r[1] is None:
            # note info
            new_midi_note = n[1]
            volume = n[3]
            if new_midi_note == previous_note:
                pass
            else:
                if previous_note is not None:
                    midi_data.append((previous_note_start_idx * SEGMENT_LENGTH / 1000.0, previous_note,
                                      (idx - previous_note_start_idx) * SEGMENT_LENGTH / 1000.0, previous_volume))
                previous_note = new_midi_note
                previous_volume = volume
                previous_note_start_idx = idx
        else:
            # rest info
            if previous_note is not None:
                midi_data.append((previous_note_start_idx * SEGMENT_LENGTH / 1000.0, previous_note,
                                  (idx - previous_note_start_idx) * SEGMENT_LENGTH / 1000.0, previous_volume))
            previous_note = None
            previous_note_start_idx = 0
            previous_volume = 0

    return midi_data


def plot(fig, data):
    times = [el[0] for el in data]
    midi_notes = [el[1] for el in data]
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=midi_notes,
                             mode='markers',
                             name='raw'))
    return fig


def plot_filtered(fig, data):
    filtered_data, rest_data = split_notes_rests(data)
    times = [el[0] for el in filtered_data]
    midi_notes = [el[1] for el in filtered_data]
    rests = [None if el[1] is None else 0 for el in rest_data]
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=midi_notes,
                             mode='markers',
                             name=f'confidence > {MIN_CONFIDENCE}'))
    fig.add_trace(go.Scatter(x=times, y=rests,
                             mode='markers',
                             name='rests'))
    return fig


def plot_simplified(simplified_data):
    times = [el[0] for el in simplified_data]
    notes = [el[1] for el in simplified_data]
    durs = [el[2] for el in simplified_data]
    vols = [el[3] for el in simplified_data]
    timeline_data = []
    for t, n, d, v in zip(times, notes, durs, vols):
        timeline_data.append(dict(Trackname="sax", Start=t, Finish=t + d, MidiNote=n, Volume=v))
    fig = go.Figure()
    for idx, t in enumerate(timeline_data):
        if idx % 10 == 0:
            print(f"{idx} of {len(timeline_data)}")
        fig.add_shape(type="rect",
                      x0=t['Start'],
                      y0=t['MidiNote'] - 0.45,
                      x1=t['Finish'],
                      y1=t['MidiNote'] + 0.45,
                      line=dict(
                          color=COLOR_EDGES[0],
                          width=2
                      ),
                      fillcolor=COLOR[t['Volume'] % 11],
                      name=t['Trackname']
                      )
    fig.update_xaxes(range=[0, timeline_data[-1]['Finish'] + 100])
    fig.update_yaxes(range=[0, 127])
    fig.show()


def dbamp(value):
    return 10 ** (value * 0.05)


def extract_amp(audio, samplerate):
    mono = audio.sum(axis=1) / 2
    scaled_mono = mono / (2**16)
    seconds = mono.size/samplerate
    segments = seconds * 1000 / SEGMENT_LENGTH
    chunks = np.array_split(scaled_mono, segments)
    dbs = [20 * math.log10(math.sqrt(statistics.mean(chunk ** 2))) for chunk in chunks]
    amps = [dbamp(el) for el in dbs]
    rescaled_amps = [round_half_up(el) for el in np.interp(amps, (min(amps), max(amps)), (60, 120))]
    return rescaled_amps


def main():
    own_path = pathlib.Path(sys.argv[0]).parent
    data = []
    if not own_path.joinpath("outputs/1.csv").exists():
        sr, audio = wavfile.read(own_path.joinpath("inputs/1.wav"))
        time, frequency, confidence, activation = crepe.predict(audio, sr, center=False, step_size=SEGMENT_LENGTH, viterbi=True)

        amps = extract_amp(audio, sr)

        with open(own_path.joinpath("outputs/1.csv"), "w") as f:
            for el in zip(time, frequency, confidence, amps*127):
                t = el[0]
                fr = el[1]
                c = el[2]
                v = el[3]
                f.write(f"{t}, {cpsmidi(fr)}, {c}, {v}\n")
                data.append((t, cpsmidi(fr), c, v))
    else:
        with open(own_path.joinpath("outputs/1.csv"), "r") as f:
            lines = f.readlines()
            for line in lines:
                split_line = line.split(",")
                data.append((float(split_line[0]), float(split_line[1]), float(split_line[2]), float(split_line[3])))

    fig = plot(None, data)
    fig = plot_filtered(fig, data)
    fig.show()

    simplified_data = simplified(data)
    mf = MIDIFile(1)
    track = 0
    time = 0
    mf.addTrackName(track, time, "Sax")
    tempo = 120
    mf.addTempo(track, time, tempo)
    channel = 0
    volume = 120
    for i in simplified_data:
        tempo_mult = tempo / 60
        t, n, d = i[0] * tempo_mult, i[1], i[2] * tempo_mult
        mf.addNote(track, channel, int(n), t, d, volume)

    print("Saving midi file")
    with own_path.joinpath("outputs/output.mid").open('wb') as outf:
        mf.writeFile(outf)
    print("Done.")


if __name__ == "__main__":
    main()
