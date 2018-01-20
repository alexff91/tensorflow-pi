from skimage import transform
import cv2
import os
import skimage
from skimage import io
from sklearn import cluster, datasets
import pafy
import pickle


def download_youtube(fileurl, name):
    '''fileurl = URL of the (youtube) video to be downloaded
    name = name video will be saved to on computer'''
    ### https://pythonhosted.org/pafy/#stream-methods and https://pypi.python.org/pypi/pafy
    v = pafy.new(fileurl)  ### uses URL provided to start to pafy instance
    s = v.getbest()  ### gets best resolution for video at link
    print("Size is %s" % s.get_filesize())  ### provides filesize
    fi = s.download(name + '.mp4')  ### downloads video file to the videos folder
    return fi


def load_frames(videofile, name, tstart, save='yes'):
    '''Function takes downloaded videos and cuts out four second clips of video to save and use for model training.
    videofile = filepath to the video to read in
    name = name of file (for saving as pickle
    tstart = array of start times (in seconds) in video to be scraped (will scrape four seconds following the tstart time given)
    '''
    ###  loads in video as a sequence using VideoCapture function
    vidcap = cv2.VideoCapture(videofile)  ### open the video file to begin reading frames
    success, image = vidcap.read()  ### reads in the first frame
    count = 0  ### starts counter at zero
    success = True  ### starts "sucess" flag at True

    while success:  ### while success == True
        success, img = vidcap.read()  ### if success is still true, attempt to read in next frame from vidcap video import
        count += 1  ### increase count
        if count in tstart:
            frames = []  ### frames will be the individual images and frames_resh will be the "processed" ones
            for j in range(0, 99):  ### for 99 frame (four second segments)
                ### conversion from RGB to grayscale image to reduce data
                success, img = vidcap.read()
                ### ref for above: https://www.safaribooksonline.com/library/view/programming-computer-vision/9781449341916/ch06.html

                tmp = skimage.color.rgb2gray(array(img))  ### grayscale image
                tmp = skimage.transform.downscale_local_mean(tmp, (3, 3))  ### downsample image
                frames.append(tmp)  ### add frame to temporary array

            count += 99  ### add to count for the frames we've just cycled through
            print
            count, tstart, name + str(count)  ### print check
            pickle.dump(frames, open(name + str(count) + '.pkl', "wb"))  ### save all frames to a pickle

    return frames


if __name__ == '__main__':
    ### Video 1 - a compilation
    download_youtube('https://youtu.be/k5KkEznW3tc', 'comp1')
    ###tstart  = array of start times, converted from minutes to seconds
    tstart = array(
        [6, 44, 1 * 60 + 17, 1 * 60 + 36, 1 * 60 + 59, 2 * 60 + 19, 2 * 60 + 51, 3 * 60 + 54, 4 * 60 + 11, 4 * 60 + 26,
         4 * 60 + 47, 5 * 60 + 6, 6 * 60 + 30]) * 30  ## tlen = (7*60+23)*30
    print(tstart)
    frames = load_frames('comp1.mp4', 'comp1', tstart)
