from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from music21 import *
import pygame
from io import BytesIO
import io
import threading
import sys
import mxm.midifile

####mycode
freq = 44100    # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2    # 1 is mono, 2 is stereo
buffer = 1024   # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)

def loadfile(xmlfilenameindirectorywquote):
    thisDir = common.getSourceFilePath() / 'musicxml'
    testFp = thisDir / xmlfilenameindirectorywquote
    c = converter.parse(testFp) #, forceSource=True)
    return c

def iterater(ms21file):
    for el in ms21file.iter.notes:
        t = stream.Stream()
        t.append(el)
        u = midi.translate.streamToMidiFile(t)
        print(u)
        #return u

ms21file1 = loadfile('test/marynotes/1a.xml')
ms21file2 = loadfile('test/marynotes/2g.xml')
ms21file3 = loadfile('test/marynotes/3f.xml')
ms21file4 = loadfile('test/marynotes/4g.xml')
ms21file5 = loadfile('test/marynotes/5a.xml')
ms21file6 = loadfile('test/marynotes/6a.xml')
ms21file7 = loadfile('test/marynotes/7a.xml')
ms21file8 = loadfile('test/marynotes/8g.xml')
ms21file9 = loadfile('test/marynotes/9g.xml')
ms21file10 = loadfile('test/marynotes/10g.xml')
ms21file11 = loadfile('test/marynotes/11a.xml')
ms21file12 = loadfile('test/marynotes/12c.xml')
ms21file13 = loadfile('test/marynotes/13c.xml')
#ms21file.show() #for veiwer reference
#midifile = iterater(ms21file)

midifile1 = midi.translate.streamToMidiFile(ms21file1)
midistring1 = midifile1.writestr()
#print(midistring)
stringfile1 = io.BytesIO(midistring1)
music_file1 = stringfile1

midifile2 = midi.translate.streamToMidiFile(ms21file2)
midistring2 = midifile2.writestr()
#print(midistring)
stringfile2 = io.BytesIO(midistring2)
music_file2 = stringfile2

midifile3 = midi.translate.streamToMidiFile(ms21file3)
midistring3 = midifile3.writestr()
#print(midistring)
stringfile3 = io.BytesIO(midistring3)
music_file3 = stringfile3

midifile4 = midi.translate.streamToMidiFile(ms21file4)
midistring4 = midifile4.writestr()
#print(midistring)
stringfile4 = io.BytesIO(midistring4)
music_file4 = stringfile4

midifile5 = midi.translate.streamToMidiFile(ms21file5)
midistring5 = midifile5.writestr()
#print(midistring)
stringfile5 = io.BytesIO(midistring5)
music_file5 = stringfile5

midifile6 = midi.translate.streamToMidiFile(ms21file6)
midistring6 = midifile6.writestr()
#print(midistring)
stringfile6 = io.BytesIO(midistring6)
music_file6 = stringfile6

midifile7 = midi.translate.streamToMidiFile(ms21file7)
midistring7 = midifile7.writestr()
#print(midistring)
stringfile7 = io.BytesIO(midistring7)
music_file7 = stringfile7

midifile8 = midi.translate.streamToMidiFile(ms21file8)
midistring8 = midifile8.writestr()
#print(midistring)
stringfile8 = io.BytesIO(midistring8)
music_file8 = stringfile8

midifile9 = midi.translate.streamToMidiFile(ms21file9)
midistring9 = midifile9.writestr()
#print(midistring)
stringfile9 = io.BytesIO(midistring9)
music_file9 = stringfile9

midifile10 = midi.translate.streamToMidiFile(ms21file10)
midistring10 = midifile10.writestr()
#print(midistring)
stringfile10 = io.BytesIO(midistring10)
music_file10 = stringfile10

midifile11 = midi.translate.streamToMidiFile(ms21file11)
midistring11 = midifile11.writestr()
#print(midistring)
stringfile11 = io.BytesIO(midistring11)
music_file11 = stringfile11

midifile12 = midi.translate.streamToMidiFile(ms21file12)
midistring12 = midifile12.writestr()
#print(midistring)
stringfile12 = io.BytesIO(midistring12)
music_file12 = stringfile12

midifile13 = midi.translate.streamToMidiFile(ms21file13)
midistring13 = midifile13.writestr()
#print(midistring)
stringfile13 = io.BytesIO(midistring13)
music_file13 = stringfile13


def play_music(music_file):
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print("Music file %s loaded!" % music_file)
    except pygame.error:
        print ("File %s not found! (%s)" % (music_file, pygame.get_error()))
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(0)

####~~~~
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# initialize the frame counters and the total number of blinks
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# loop over frames from the video stream

# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

COUNTER = 0
TOTAL = 0
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                if TOTAL == 1:
                    play_music(stringfile1)
                if TOTAL == 2:
                    play_music(stringfile2)
                if TOTAL == 3:
                    play_music(stringfile3)
                if TOTAL == 4:
                    play_music(stringfile4)
                if TOTAL == 5:
                    play_music(stringfile5)
                if TOTAL == 6:
                    play_music(stringfile6)
                if TOTAL == 7:
                    play_music(stringfile7)
                if TOTAL == 8:
                    play_music(stringfile8)
                if TOTAL == 9:
                    play_music(stringfile9)
                if TOTAL == 10:
                    play_music(stringfile10)
                if TOTAL == 11:
                    play_music(stringfile11)
                if TOTAL == 12:
                    play_music(stringfile12)
                if TOTAL == 13:
                    play_music(stringfile13)
                    
            # reset the eye frame counter
            COUNTER = 0

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blink/MIDIStep: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "thresh: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break



# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
