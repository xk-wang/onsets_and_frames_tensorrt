#include "MidiFile.h"
#include "Options.h"
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;
using namespace smf;

// ./midi txt_path midi_path
int main(int argc, char** argv) {
    if(argc!=3){
        cout << "wrong paramerters" << endl;
        return 0;
    }
    // 读取txt
    ifstream txtfile (argv[1]);
    if(!txtfile.is_open()){
        cout << "open file failed" << endl;
        return 0;
    }

    string line;
    vector<vector<float>> pianroll;
    float d;
    while (getline (txtfile, line)) {
        stringstream ss;
        ss << line;
        vector<float> row;
        while(ss >> d) {
            row.push_back(d);
            if (ss.peek() == '\t') {
                ss.ignore();
            }
        }
        pianroll.push_back(row);
    }
    txtfile.close();

    int tpq = 120;
    double defaultTempo = 120.0;
	double secondsPerTick = 60.0 / (defaultTempo * tpq);

    MidiFile midifile;
    int track   = 0;
    int channel = 0;
    int instr   = 0;
    midifile.setTPQ(tpq);
    midifile.addTimbre(track, 0, channel, instr);

    for (const auto& note: pianroll) {
        int starttick = int(note[0]/secondsPerTick);
        int endtick   = int(note[1]/secondsPerTick);
        int key       = int(note[2]);
        int velocity  = int(note[3]);
        midifile.addNoteOn (track, starttick, channel, key, velocity);
        midifile.addNoteOff(track, endtick,   channel, key);
    }
    midifile.sortTracks();  // Need to sort tracks since added events are
                            // appended to track in random tick order.
    midifile.write(argv[2]);
    return 0;
}