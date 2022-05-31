from utils import get_paths, get_data, audio_paths


print('SELECTING DATA AND SPLITTING')
split_options = ['train', 'valid', 'test']
for option in split_options:
    print(option)
    files = get_paths(option, audio_paths)
    get_data(files, option)
print('Processing Done')



