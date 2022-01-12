#include <iostream>
#include <vector>
#include <sndfile.h>
#include <string.h>
#define BUFFER_LEN ( 1<<14 )

// read wav path to test libsndfile
int main(int argc, char* argv[]){
    const char* filepath = argv[1];
    SNDFILE *infile = NULL;
    SF_INFO in_sfinfo, out_sfinfo, left_sfinfo, right_sfinfo;
    memset(&in_sfinfo, 0, sizeof(in_sfinfo));
    if((infile=sf_open(filepath, SFM_READ, &in_sfinfo))==NULL){
        std::cout << "Not able to open output file " << filepath << std::endl;
        sf_close(infile);
        return 1;
    }

    std::cout << "file frames: " << in_sfinfo.frames << std::endl
              << "the sample rate: " << in_sfinfo.samplerate << std::endl
              << "channels: " << in_sfinfo.channels << std::endl
              << "format: " << in_sfinfo.format << std::endl
              << "sections: " << in_sfinfo.sections << std::endl
              << "seekable: " << in_sfinfo.seekable << std::endl;
    memcpy(&out_sfinfo, &in_sfinfo, sizeof(in_sfinfo));
    memcpy(&left_sfinfo, &in_sfinfo, sizeof(in_sfinfo));
    memcpy(&right_sfinfo, &in_sfinfo, sizeof(in_sfinfo));
    left_sfinfo.channels =  right_sfinfo.channels = 1;
    std::cout << left_sfinfo.frames<< " " << right_sfinfo.frames << std::endl;

    if(in_sfinfo.format!= (SF_FORMAT_WAV | SF_FORMAT_PCM_16)){ // 前者是主格式 后者是编码类型 
        std::cout << "the input file is not wav format!" << std::endl;
        return 1;
    }
    
    static short data[BUFFER_LEN];
    int frames, readcount;
    frames = BUFFER_LEN / in_sfinfo.channels;
    readcount = frames;
    int total_frames = 0;
    while(readcount >0){
        readcount = (int) sf_readf_short(infile, data, frames);
        total_frames += readcount;
    }
    std::cout << "the total read audio frames: " << total_frames << std::endl;
    
    sf_seek(infile, 0, SEEK_SET);
    short* total = new short[out_sfinfo.frames*2];

    readcount = (int) sf_readf_short(infile, total, in_sfinfo.frames);
    std::cout << "the total read count: " << readcount << std::endl;
    sf_close(infile);
    short* left = new short[left_sfinfo.frames];
    short* right = new short[right_sfinfo.frames];
    for(int i=0;i<out_sfinfo.frames;++i){
        left[i] = total[2*i];
        right[i] = total[2*i+1];
    }

    SNDFILE *outputfile = NULL;
    std::cout << "preparing to write frames: " << out_sfinfo.frames << std::endl;

    // 写之后参数会被重置，因此变得不可用
    if((outputfile = sf_open("output.wav", SFM_WRITE, &out_sfinfo)) == NULL){
        std::cout << "Not able to open output file" << std::endl;
		return 1;
    }
    int count = (int)sf_writef_short(outputfile, total, in_sfinfo.frames);
    std::cout << "the output count: " << count << std::endl;
    sf_close(outputfile);
    delete [] total;

    SNDFILE *leftfile = NULL;
    if((leftfile = sf_open ("left.wav", SFM_WRITE, &left_sfinfo)) == NULL)
	{	std::cout << "Not able to open output file" << std::endl;
		return 1;
	}

    count = sf_writef_short(leftfile, left, in_sfinfo.frames);
    std::cout << "the output count: " << count << std::endl;
    sf_close(leftfile);
    delete [] left;

    SNDFILE *rightfile = NULL;
    if((rightfile = sf_open("right.wav", SFM_WRITE, &right_sfinfo)) == NULL)
    {
        std::cout << "Not able to open output file" << std::endl;
		return 1;
    }
    count = sf_writef_short(rightfile, right, in_sfinfo.frames);
    std::cout << "the output count: " << count << std::endl;

    sf_close(rightfile);
    delete []right;

    return 0;
}