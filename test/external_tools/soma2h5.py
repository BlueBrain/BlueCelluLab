import os, sys
import binreports
import h5py
import progressbar as pb

#br = binreports.BinReport(os.path.join("/bgscratch/bbp/ebmuller/simulations/8papers/runs/01.09.11/centerSurround/e1Hz_i2Hz_nominis_I4_Ge80_oldNMDA","soma.bbp"))

br = binreports.BinReport("soma.bbp")

h5 = h5py.File("soma.h5", 'w')

gids = br.cell_gids()
num_frames = br.num_frames()
frame_chunk = 500

print "Creating h5 data sets ..."
d = [h5.create_dataset('a%d' % gid, (num_frames,), "<f4", chunks=(frame_chunk,),compression='gzip',compression_opts=5, track_times=None) for gid in gids]

widgets = ['soma to h5: ', pb.Percentage(), ' ', pb.Bar(),
           ' ', pb.ETA()]
pbar = pb.ProgressBar(widgets=widgets, maxval=num_frames/frame_chunk).start()

print "Number of frames: %d" % num_frames
print "Number of chunks %d of size %d" % (num_frames/frame_chunk, frame_chunk)


for chunk_count, frame_begin in enumerate(xrange(0,num_frames,frame_chunk)):
    frames = br.read_frames(frame_begin, frame_chunk)
    for j in xrange(len(gids)):
        d[j][frame_begin:frame_begin+frame_chunk]=frames[:,j]
    pbar.update(chunk_count+1)

pbar.finish()
print "Writing/closing h5 file ..."
h5.close()
#h5.close()


