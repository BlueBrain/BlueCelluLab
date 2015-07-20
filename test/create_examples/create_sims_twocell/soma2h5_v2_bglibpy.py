"""Soma h5 v2 adapted for bglibpy"""

import pybinreports
import h5py
import progressbar as pb
import argparse


def main(bbp_filename, compression_lev=9, frame_chunk=500):
    """Main"""

    # Setup filenames
    h5_filename = bbp_filename[:-3] + 'h5'

    # Create the binary report object
    br = pybinreports.BinReport(bbp_filename)

    # Create the h5 file
    h5 = h5py.File(h5_filename, 'w', libver='latest')
    h5.attrs.create('version', 1.0)

    # Retrieve the gids of the cells in the report
    gids = br.cell_gids()
    gids_count = len(gids)

    # Retrieve the number of frames in the report
    num_frames = br.num_frames()

    # Read mapping
    mapping = br.read_mapping()

    # Read extramapping
    # TODO Check if the name extramapping is appropriate
    extramapping = br.read_extramapping()

    # Compute the boundaries of the mapping array
    dim = len(extramapping)
    bstart = [0] * dim
    bstop = [0] * dim
    bstop[0] = bstart[0] + extramapping[0]
    i = 1
    while i < dim:
        bstart[i] = bstop[i - 1]
        bstop[i] = bstart[i] + extramapping[i]
        i = i + 1

    # Create a new h5 dataset for each cell in the report
    # print 'Creating h5 data sets ...'
    #groups = [h5.create_group('a%d' % gid) for gid in gids]
    # for idx, group in enumerate(groups):
    #   group.create_dataset('mapping', data=mapping[bstart[idx]:bstop[idx]])
    #   group.create_dataset('data', (num_frames, extramapping[idx]), '<f4', chunks=(frame_chunk,extramapping[idx]), compression='gzip', compression_opts=compression_lev)
    #   group.attrs['GID'] = 'a%d' % gids[idx]

    # Create a progress bar for the conversion
    widgets = [
        'Creating h5 data sets: ',
        pb.Percentage(),
        ' ',
        pb.Bar(),
        ' ',
        pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=gids_count).start()

    for idx, gid in enumerate(gids):
        h5.create_dataset(
            'a%d' %
            (gid,
             ),
            (num_frames,
             extramapping[idx]),
            '<f4',
            chunks=(
                frame_chunk,
                extramapping[idx]),
            compression='gzip',
            compression_opts=compression_lev)
        pbar.update(idx)
    pbar.finish()

    # Create a progress bar for the conversion
    widgets = ['bbp to h5: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(
        widgets=widgets,
        maxval=gids_count *
        num_frames /
        frame_chunk).start()

    # Print info
    print 'Number of frames: %d' % (num_frames,)
    print 'Number of chunks %d of size %d' % (num_frames / frame_chunk, frame_chunk)
    print 'Compression level %d' % (compression_lev,)

    # Convert
    for chunk_count, frame_begin in enumerate(xrange(0, num_frames, frame_chunk)):
        # Read the current frame
        frames = br.read_frames(frame_begin, frame_chunk)

        # Set the values in the h5 file for each cell
        for j in xrange(gids_count):
            h5['/a%d' %
               (gids[j])][frame_begin:frame_begin +
                          frame_chunk] = frames[:, bstart[j]:bstop[j]]
            # Update the progress bar
            pbar.update(gids_count * chunk_count + j + 1)

    # Finish
    pbar.finish()
    print 'Writing/closing h5 file ...'
    h5.close()


if __name__ == '__main__':
    # Setup command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chunk', type=int, help='frame chunk size')
    parser.add_argument(
        '-C',
        '--compression',
        type=int,
        help='Compression level')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='increase output verbosity')
    parser.add_argument('filename', help='bbp file to convert')

    # Parse the command line
    args = parser.parse_args()

    if args.verbose:
        print 'verbosity turned on'

    if args.chunk:
        frame_chunk = args.chunk
    else:
        frame_chunk = 5000

    if 0 < args.compression < 10:
        compression_lev = args.compression
    else:
        compression_lev = 9

    main(args.filenam, compression_lev=compression_lev, frame_chunk=frame_chunk)
