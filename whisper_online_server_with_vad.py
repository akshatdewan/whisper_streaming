#!/usr/bin/env python3
from whisper_online import *
from socket_stream import SocketStream
import sys
import argparse
import os
from voice_activity_controller import VoiceActivityController

parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)


# options from whisper_online
# TODO: code repetition

parser.add_argument('--min-chunk-size', type=float, default=1.0, help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.')
parser.add_argument('--model', type=str, default='large-v2', choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large".split(","),help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.")
parser.add_argument('--model_cache_dir', type=str, default=None, help="Overriding the default model cache dir where models downloaded from the hub are saved")
parser.add_argument('--model_dir', type=str, default=None, help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
parser.add_argument('--lan', '--language', type=str, default='en', help="Language code for transcription, e.g. en,de,cs.")
parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],help="Transcribe or translate.")
parser.add_argument('--backend', type=str, default="faster-whisper", choices=["faster-whisper", "whisper_timestamped"],help='Load only this backend for Whisper processing.')
parser.add_argument('--vad', action="store_true", default=False, help='Use VAD = voice activity detection, with the default parameters.')
parser.add_argument('--logfile', type=argparse.FileType('w',encoding='utf-8',errors='strict'), default=sys.stderr, help='Log file')
args = parser.parse_args()


# setting whisper object by args 

SAMPLING_RATE = 16000

size = args.model
language = args.lan
logfile = args.logfile

t = time.time()
print(f"Loading Whisper {size} model for {language}...",file=sys.stdout,end=" ",flush=True)

if args.backend == "faster-whisper":
    from faster_whisper import WhisperModel
    asr_cls = FasterWhisperASR
else:
    import whisper
    import whisper_timestamped
#    from whisper_timestamped_model import WhisperTimestampedASR
    asr_cls = WhisperTimestampedASR

asr = asr_cls(modelsize=size, lan=language, cache_dir=args.model_cache_dir, model_dir=args.model_dir, logfile=args.logfile)

if args.task == "translate":
    asr.set_translate_task()
    tgt_language = "en"
else:
    tgt_language = language

e = time.time()
print(f"done. It took {round(e-t,2)} seconds.",file=sys.stdout)

if args.vad:
    print("setting VAD filter",file=sys.stdout)
    asr.use_vad()


min_chunk = args.min_chunk_size
online = OnlineASRProcessor(asr,create_tokenizer(tgt_language),logfile=args.logfile)



demo_audio_path = "cs-maji-2.16k.wav"
if os.path.exists(demo_audio_path):
    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(demo_audio_path,0,1)

    # TODO: it should be tested whether it's meaningful
    # warm up the ASR, because the very first transcribe takes much more time than the other
    asr.transcribe(a)
else:
    print("Whisper is not warmed up",file=sys.stdout)




######### Server objects

import line_packet
import socket

import logging


class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        r = self.conn.recv(self.PACKET_SIZE)
        return r


import io
import soundfile

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class ServerProcessor:
    #AD
    def __init__(self, ss, online_asr_proc, min_chunk):
        self.socket_stream = ss 
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.last_end = None
    
    def format_output_transcript(self,o):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print("%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stdout)
            return "%1.0f %1.0f %s" % (beg,end,o[2])
        else:
            print(o,file=sys.stdout,flush=True)
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            #AD
            pass
            #self.connection.send(msg)

    def process(self):
        # handle one client connection
        self.online_asr_proc.init()
        min_sample_length =  self.min_chunk * SAMPLING_RATE
        online = self.online_asr_proc
        use_vad_result = True
        vad = VoiceActivityController(use_vad_result = use_vad_result)

        complete_text = ''
        final_processing_pending = False
        out = []
        out_len = 0
        for iter in vad.detect_user_speech(self.socket_stream):
            raw_bytes=  iter[0]
            is_final =  iter[1]
        
            if  raw_bytes:
                sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
                audio, _ = librosa.load(sf,sr=SAMPLING_RATE)
                out.append(audio)
                out_len += len(audio)
        
            if (is_final or out_len >= min_sample_length) and out_len>0:
                a = np.concatenate(out)
                online.insert_audio_chunk(a)    
                
            if out_len > min_sample_length:
                o = online.process_iter()
                #print('-----'*10)
                #complete_text = complete_text + o[2]
                if len(o[2]) > 0:
                    sys.stdout.write(" {}".format(o[2].strip()))
                    sys.stdout.flush()
                #print('PARTIAL - '+ complete_text) # do something with current partial output
                #print('-----'*10)     
                out = []
                out_len = 0   
        
            if is_final:
                o = online.finish()
                online.init()   
                # final_processing_pending = False         
                #print('-----'*10)
                complete_text = complete_text + o[2]
                if not o[2] == " ":
                    sys.stdout.write(" {}".format(o[2].strip()))
                    sys.stdout.flush()
                #print('FINAL - '+ complete_text) # do something with current partial output
                #print('-----'*10)   
                out = []
            

# Start logging.
level = logging.INFO
logging.basicConfig(level=level, filename=logfile.name, format='whisper-server-%(levelname)s: %(message)s')

# server loop

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((args.host, args.port))
    s.listen(1)
    logging.info('INFO: Listening on'+str((args.host, args.port)))
    while True:
        conn, addr = s.accept()
        logging.info('INFO: Connected to client on {}'.format(addr))
        #connection = Connection(conn)
        socket_stream = SocketStream(conn)
        #proc = ServerProcessor(connection, online, min_chunk)
        proc = ServerProcessor(socket_stream, online, min_chunk)
        proc.process()
        conn.close()
        logging.info('INFO: Connection to client closed')
logging.info('INFO: Connection closed, terminating.')
