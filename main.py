from flask import Flask, render_template, Response, request,jsonify
from camera import VideoCamera
from music_classifier.music_classifier import get_music_from_func
from music_classifier.model import get_music_from_model

app = Flask(__name__)
source_flag ="func"   # change this to func or music_model
video_feed_paused = False
last_frame = None

@app.route('/')
def index():
    return render_template('index_4.html')

def gen(camera):
    global last_frame
    while True:
        if not video_feed_paused:
            frame = camera.get_frame()
            last_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n\r\n')



@app.route('/pause_feed')
def pause_feed():
    global video_feed_paused
    video_feed_paused = True
    return jsonify({"status": "paused"})

@app.route('/resume_feed')
def resume_feed():
    global video_feed_paused
    video_feed_paused = False
    return jsonify({"status": "resumed"})

@app.route('/get_last_emotion')
def get_last_emotion():
    camera = VideoCamera()
    last_detected_emotion = camera.get_last_detected_emotion()
    return last_detected_emotion if last_detected_emotion else "None"

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_music')
def get_music():
    emotion = request.args.get('emotion', 'None')
    if(source_flag=="func"):
        music_tracks = get_music_from_func(emotion,True)
    elif(source_flag=="music_model"):
        music_tracks=get_music_from_model(emotion)

    return jsonify(music_tracks)


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
