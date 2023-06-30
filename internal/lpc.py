import numpy as np
import os
import parselmouth

### LPC coefficient calculation based on temporary speech signals at pre-defined measurement nodes

### Part of the VTSim Package

def find_formants(node_signals, max_number_of_formants, sim_name, nodes, sim_output_path):
    header_file = np.loadtxt(sim_output_path+'\\'+str(sim_name)+'.txt', delimiter=': ',dtype='str')
    header = {row[0]:row[1] for row in header_file}
    fs = int(header['Sim Frequency'])

    node_positions = nodes[1:] ### Don't need formant values for source node

    node_index = 0
    speech_qualities = []
    for data in node_signals:
        print('Node Position:', node_positions[node_index])
        snd = parselmouth.Sound(data, fs)
        pitch = snd.to_pitch(time_step=0.025) # pitch track
        form = snd.to_formant_burg(time_step=0.025,max_number_of_formants=max_number_of_formants, maximum_formant = 5500, pre_emphasis_from=50.0)
        times = pitch.ts()
        formants = []
        bandwidths = []
        for dt in times:
            form1 = form.get_value_at_time(1,dt)
            band1 = form.get_bandwidth_at_time(1,dt)
            form2 = form.get_value_at_time(2,dt)
            band2 = form.get_bandwidth_at_time(2,dt)
            form3 = form.get_value_at_time(3,dt)
            band3 = form.get_bandwidth_at_time(3,dt)
            form4 = form.get_value_at_time(4,dt)
            band4 = form.get_bandwidth_at_time(4,dt)
            if max_number_of_formants == 5:
                form5 = form.get_value_at_time(5,dt)
                band5 = form.get_bandwidth_at_time(5,dt)
                formants.append([form1,form2,form3,form4,form5])
                bandwidths.append([band1,band2,band3,band4,band5])
            else:
                formants.append([form1,form2,form3,form4])
                bandwidths.append([band1,band2,band3,band4])
        formants = np.average(np.array(formants)[:-1], axis=0).astype(int)
        bandwidths = np.average(np.array(bandwidths)[:-1], axis=0).astype(int)
        # for i in range(formants.shape[1]):
        #     print(round(np.average(formants[:,i])))
        #     print(round(np.average(bandwidths[:,i])))
        print('Formants:',formants)
        print('Bandwidths:',bandwidths)
        node_index+=1
        speech_qualities.append([formants, bandwidths])

    np.savetxt(sim_output_path+'\\formant_frequencies.txt',np.asarray(speech_qualities)[:,0,:])
    np.savetxt(sim_output_path+'\\formant_bandwidths.txt',np.asarray(speech_qualities)[:,1,:])
    return speech_qualities
