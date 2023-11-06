
tum_path = "./data/SLAM_DATA/Arthroscopy/poses_gt.txt"

lines = []
with open(tum_path, "r") as handle:
    lines = handle.readlines()

frame_idx = 1
with open(tum_path, "w") as handle:
    for line in lines:
        if line[0] == '#':
            handle.write(line)
            continue
        line = line.replace(line[:line.find(' ')], str(frame_idx))
        line = line.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
        handle.write(line)
        frame_idx += 1