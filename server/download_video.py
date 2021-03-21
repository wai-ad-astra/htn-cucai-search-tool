from pytube import YouTube
import os
from itertools import count
import cv2

def get_video(video_name='video', video_link=None):
  if video_name not in (name[:name.rindex('.')] for name in os.listdir() if '.' in name):
    if not video_link:  # use in colab, not production
      video_link = input('video link to download: ')
    YouTube(video_link).streams.first().download()
    print(f'video downloaded as {video_name}')
    
    # rename downloaded vid to video_name
    tmp_video = next(name for name in os.listdir() if name.endswith('mp4'))
    video = f"{video_name}{tmp_video[tmp_video.rindex('.'):]}"
    os.rename(tmp_video, video)
  else:
    video = next(name for name in os.listdir() if name.startswith(video_name))
    print(f'{video} found!')
  return video
# INPUT_VID = 'video.mp4' if 'video.mp4' in os.listdir() else '':

#get_video('',"https://www.youtube.com/watch?v=F12PJgyVKyA")
VIDEO_NAME = get_video(video_name='petting_zoo', video_link='https://www.youtube.com/watch?v=AVPuJMtzrCw')
IMAGE_OUTPUT_PATH = os.path.join('.', 'image_output')
SAMPLING_RATE = 1  # in fps
FRAMES_PER_SAMPLE = 1000 // SAMPLING_RATE

def extractImages(pathIn=VIDEO_NAME, pathOut=IMAGE_OUTPUT_PATH):
    #os.makedirs(IMAGE_OUTPUT_PATH)

    vidcap = cv2.VideoCapture(pathIn)
    counter = count()  # defaults to start=0, step=1

    success, image = vidcap.read()
    cur_count = next(counter)
    while success:
        cv2.imwrite(str(os.path.join(pathOut, f'frame{str(cur_count).zfill(6)}.jpg')), image)     # save frame as JPEG file
        if cur_count % 10 == 0:
          print (f'at sample {cur_count}')

        cur_count = next(counter)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, cur_count * FRAMES_PER_SAMPLE)  # POS_MSEC is position in miliseconds
        success, image = vidcap.read()
       
def getImgs(name,link):
    videoname = get_video(name,link)
    extractImages(videoname,'')

#!rm -r ./image_output
