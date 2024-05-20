from torchaudio.kaldi_io import read_mat_scp
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile as wav
import glob, os, sys

if __name__=="__main__":
    i=0
    unit=50  ## time shift
    outdir=sys.argv[1]
    feat_dim=int(sys.argv[2])
    file=outdir+"/feats.scp"
    #outdir="dt05_orig_clean/"
    #file="dt05_orig_clean/feats.scp"
    for key, mat in read_mat_scp(file):
        fbank_data = mat.cpu().numpy().reshape((-1,feat_dim))
        print(fbank_data.shape)
        print(int(int(fbank_data.shape[0])/unit))
        for index in range(1, int(int(fbank_data.shape[0])/unit)):
            start = (index-1)*unit
            end = (index)*unit
            #print(fbank_data[start:end,0:1].shape)
            #librosa.display.specshow(fbank_data[start:end, 0:10], x_axis='time')
            librosa.display.specshow(fbank_data[start:end], x_axis='time', y_axis='mel', fmax=8000)
            title=key+'_'+str(start*10)+"-"+str(end*10)+' ms' # each frame contains 10 ms
            plt.title(str(title))
            plt.show()
            if i == 0:
                plt.clim(0,30)
                plt.colorbar()
            i=i+1
            output=outdir+'/'+title+"-bin0-40.png"
            print(output)
            plt.savefig(output)

