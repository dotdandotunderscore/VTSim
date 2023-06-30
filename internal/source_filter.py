import numpy as np
from scipy.fft import ifft
from scipy.io.wavfile import write
from scipy.interpolate import interp1d
import os

### Production of temporary speech signals using filtering of Liljencrants-Fant (LF) with VTTF at pre-defined nodes

### Part of the VTSim Package

def make_temp_speech_signal(vttfs, sim_name, nodes, sim_output_path):
    header_file = np.loadtxt(sim_output_path+'\\'+str(sim_name)+'.txt', delimiter=': ',dtype='str')
    header = {row[0]:row[1] for row in header_file}
    N = int(header['Sim Steps']) ### Number of time steps in simulation
    l,m,n = list(map(int,header['Source Node Location'][1:-1].split(', ')))

    ### Data for the LF model here comes from the LFmodelCalc-23-02-2015.xlsx provided to me by David. A typical LF
    ### pulse seems to be characterised as 10ms long. Therefore, the 1023 point data set for this LF data would have
    deltaT = 10E-3/1023 ### = 9.775$\mu$s
    freq = 1/deltaT ### 102.4kHz
    SAMPLE_RATE = freq
    ### So the minimum Nyquist Frequency here is
    nyquist = freq / 2 ### 51.15kHz

    internal_path = os.path.dirname(os.path.realpath(__file__))
    lf_model='full_lf1024Excel'

    node_positions = nodes[1:] ### Don't need to produce output for source node

    node_signals = []
    node_index = 1
    for vttf in vttfs:
        source = np.fft.fftshift(np.loadtxt(internal_path+'\\'+lf_model+'.txt', dtype=np.complex128))
        source /= source.max()
        source_xf = np.fft.fftshift(np.loadtxt(internal_path+'\\'+lf_model+'xf.txt', dtype=np.complex128))
        source_range = source_xf[-1]-source_xf[0]
        source_dT = source_xf[1]-source_xf[0]

        ### We need VTTF_xf and source_xf to be identical to perform filtering.
        ### VTTF_xf has a larger range than source_xf so we slice it to the same range
        ### VTTF_xf has a higher frequency than source_xf so we interpolate source_xf to match
        VTTF = np.fft.fftshift(vttf[0])
        VTTF /= VTTF.max()
        VTTF_xf = np.fft.fftshift(vttf[1])

        VTTF_range = VTTF_xf[-1]-VTTF_xf[0]
        VTTF_dT = VTTF_xf[1]-VTTF_xf[0]
        if VTTF_range > source_range:
            if VTTF_dT > source_dT:
                ### Finds the correct amount to slice VTTF by, based on the range of source_xf
                diff = VTTF_xf - source_xf[0]
                mask = np.ma.less_equal(diff, 0)
                masked_diff = np.ma.masked_array(diff, mask)
                cut_id_low = masked_diff.argmin()
                diff = VTTF_xf - source_xf[-1]
                mask = np.ma.less_equal(diff, 0)
                masked_diff = np.ma.masked_array(diff, mask)
                cut_id_high = masked_diff.argmin()

                VTTF = VTTF[cut_id_low-1:cut_id_high+1]
                VTTF_xf = VTTF_xf[cut_id_low-1:cut_id_high+1]

                ### Stretch VTTF to required size with interpolation
                ### The data in interp1d must have a larger range than what is passed to interp
                interp = interp1d(VTTF_xf, VTTF)
                VTTF = interp(source_xf)/interp(source_xf).max()
            else:
                ### Finds the correct amount to slice VTTF by, based on the range of source_xf
                diff = VTTF_xf - source_xf[0]
                mask = np.ma.less_equal(diff, 0)
                masked_diff = np.ma.masked_array(diff, mask)
                cut_id_low = masked_diff.argmin()
                diff = VTTF_xf - source_xf[-1]
                mask = np.ma.less_equal(diff, 0)
                masked_diff = np.ma.masked_array(diff, mask)
                cut_id_high = masked_diff.argmin()

                VTTF = VTTF[cut_id_low:cut_id_high]
                VTTF_xf = VTTF_xf[cut_id_low:cut_id_high]

                ### Stretch source to required size with interpolation
                ### The data in interp1d must have a larger range than what is passed to interp
                interp = interp1d(source_xf, source)
                source = interp(VTTF_xf)/interp(VTTF_xf).max()

        else:
            if VTTF_dT > source_dT:
                ### Finds the correct amount to slice source by, based on the range of VTTF_xf
                diff = source_xf - VTTF_xf[0]
                mask = np.ma.less_equal(diff, 0)
                masked_diff = np.ma.masked_array(diff, mask)
                cut_id_low = masked_diff.argmin()
                diff = source_xf - VTTF_xf[-1]
                mask = np.ma.less_equal(diff, 0)
                masked_diff = np.ma.masked_array(diff, mask)
                cut_id_high = masked_diff.argmin()

                source = source[cut_id_low:cut_id_high]
                source_xf = source_xf[cut_id_low:cut_id_high]

                ### Stretch source to required size with interpolation
                ### The data in interp1d must have a larger range than what is passed to interp
                interp = interp1d(source_xf, source)
                source = interp(VTTF_xf)/interp(VTTF_xf).max()
            else:
                ### Finds the correct amount to slice source by, based on the range of VTTF_xf
                diff = source_xf - VTTF_xf[0]
                mask = np.ma.less_equal(diff, 0)
                masked_diff = np.ma.masked_array(diff, mask)
                cut_id_low = masked_diff.argmin()
                diff = source_xf - VTTF_xf[-1]
                mask = np.ma.less_equal(diff, 0)
                masked_diff = np.ma.masked_array(diff, mask)
                cut_id_high = masked_diff.argmin()

                source = source[cut_id_low-1:cut_id_high+1]
                source_xf = source_xf[cut_id_low-1:cut_id_high+1]

                ### Stretch VTTF to required size with interpolation
                ### The data in interp1d must have a larger range than what is passed to interp
                interp = interp1d(VTTF_xf, VTTF)
                VTTF = interp(source_xf)/interp(source_xf).max()

        # if VTTF_dT < source_dT:
        #     ### Stretch source to required size with interpolation
        #     ### The data in interp1d must have a larger range than what is passed to interp
        #     interp = interp1d(source_xf, source)
        #     source = interp(VTTF_xf)/interp(VTTF_xf).max()
        # else:
        #     ### Stretch VTTF to required size with interpolation
        #     ### The data in interp1d must have a larger range than what is passed to interp
        #     interp = interp1d(VTTF_xf, VTTF)
        #     VTTF = interp(source_xf)/interp(source_xf).max()

        ### filtering in frequency domain is multiplication
        filtered = source  * VTTF

        # ### SOUND SYNTHESIS
        new_sig = ifft(np.fft.ifftshift(filtered))
        normalized_tone = np.real(new_sig)/np.abs(new_sig).max()
        normalized_sig = np.tile(normalized_tone,1000)
        node_signals.append(normalized_sig)

        write(os.getcwd()+'\\sf_notile.wav', 595825, normalized_tone)
        write(os.getcwd()+'\\sf_tiled.wav', 595825, normalized_sig)

        node_index+=1

    return node_signals
