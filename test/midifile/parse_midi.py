import pretty_midi
import os
import argparse
pretty_midi.pretty_midi.MAX_TICK=1e10

parser = argparse.ArgumentParser()
parser.add_argument('midi_path', type=str)
parser.add_argument('txt_path', type=str)

def extract_labels_from_midi(midi_file):
	midi_data = pretty_midi.PrettyMIDI(midi_file)
	outputs = []
	for instrument in midi_data.instruments:
		notes = instrument.notes
		for note in notes:
			start = note.start
			end = note.end
			pitch = note.pitch
			velocity = note.velocity
			outputs.append([start, end, pitch, velocity])
	outputs.sort(key = lambda elem: elem[0])
	return outputs

def parse_midi(args):
    midi_path = args.midi_path
    txt_path = args.txt_path
    notes = extract_labels_from_midi(midi_path)
    with open(txt_path, 'wt') as f:
        for onset, offset, pitch, velocity in notes:
            f.write('%.4f\t%.4f\t%d\t%d\n'%(onset, offset, pitch, velocity))

if __name__ == '__main__':
    parse_midi(parser.parse_args())
