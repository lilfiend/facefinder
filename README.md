# facefinder
This repo is just me learning how to do facial recognition. Its not meant for anyone besides me to use but I`ll try to keep things documented well enough that others can.

# Requirements
⋅⋅* linux probably, although I think everything could be ran on windows I haven`t tried and don`t plan to

⋅⋅* OpenCV

⋅⋅* Dlib

⋅⋅* face_recognition

⋅⋅* imutils

⋅⋅* numpy

### Optional

⋅⋅* An nvidia card that is capable of cuda compute core 3.0 +
⋅⋅* Cuda + CudaNN



## dataset.py
To start things off you need to create a dataset of images, this is the script that lets you do that. Running the script will make a preview of the video pop up, it will draw a box around your face so you can make sure that it actually detects a face in frame. Pushing `s` will save an image, `q` will quit. You can push `s` a whole bunch before quitting and it will save out a whole bunch of images. 

Sample usage:
`python dataset.py --video 0 --model cnn --output dataset/lilfiend`

#### Arguments

⋅⋅*`--video` is the video source to capture images from. `--video 0` is for local webcam, or you can use an IP camera like so: `--video http://192.168.1.5/cam_1.cgi`

⋅⋅*`--model` is the model its going to use to detect your face. It does this only so you can see it, you can chose from `--model cnn` or `--model hog`. CNN is much more accurate but if you do not have CUDA accelleration you will not be able to do it in realtime. HOG is less accurate but quite a bit faster, you still probably won`t get full framerate from your camera unless you have a good cpu.

⋅⋅*`--output` is the directory where its going to save all these images. Note later scripts pulls from the folder name so I highly recommend doing something like `--output dataset/lilfiend`

## encode.py
encode.py is the only non-realtime script so far. You point it to a folder full of folders that are full of pictures of faces. It then loops through all of them and encodes the faces.

Sample usage:
`python encode.py --dataset dataset --output cnn.pickle --model cnn`

#### Arguments

⋅⋅*`--dataset` Point this to the folder that has folders full of images. IE `--dataset faces` where faces/ contains faces/jeff, faces/joe, faces/bob, faces/steve, etc.

⋅⋅*`--output` This is the output file. I use pickle to store info so I usually name it as `cnn.pickle` or `hog.pickle` but you can name it whatever you please

⋅⋅*`model` This is the important one, even if you don`t have CUDA acceleration I`d recommend using cnn here as it is quite accurate. hog is a lot faster though so if you`re trying to get through a large dataset you might want to go that route.

## facefinder.py
This is where it all comes together. It pulls realtime video from webcam/ipcam, runs a model against it (cnn or hog) in realtime to detect the face. After it detects the face it then compares the face to the known face encodings from earlier. This is quite intensive but it works great. The lightweight alternative to cnn/hog is to use a haarcascade but they are quite terrible at detecting faces reliably and they produce a lot of false positives.

Sample usage:
`python facefinder.py --video 0 --model cnn --trained cnn.pickle`

#### Arguments

⋅⋅*`--video` is the video source. `--video 0` is for local webcam, or you can use an IP camera like so: `--video http://192.168.1.5/cam_1.cgi`

⋅⋅*`--model` is the model its going to use to detect your face, just like in `encode.py`. The big difference is that we aren`t JUST doing this so you can see the box, we do it so we can compare the detected face against the known encoded faces.

⋅⋅*`--trained` Point this to the file you output from `encode.py`.
