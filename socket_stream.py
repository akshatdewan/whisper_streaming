### socket stream

import queue
import re
import sys

import logging

import line_packet
import socket

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


class SocketStream:
    def __init__(
        self,
        conn,
        sample_rate: int = 16000,
    ):
        """
        Creates a stream of audio from the socket.

        Args:
            chunk_size: The size of each chunk of audio to read from the microphone.
            channels: The number of channels to record audio from.
            sample_rate: The sample rate to record audio at.
        """
        self.sample_rate = sample_rate
        self._chunk_size = int(self.sample_rate * 0.1)
        self._conn = conn
        self._stream = Connection(conn)
        self._open = True

    def __iter__(self):
        """
        Returns the iterator object.
        """

        return self

    def __next__(self):
        """
        Reads a chunk of audio from the microphone.
        """
        if not self._open:
            raise StopIteration

        try:
            return self._stream.non_blocking_receive_audio()
        except KeyboardInterrupt:
            raise StopIteration

    def close(self):
        """
        Closes the stream.
        """

        self._open = False

        self._conn.close()

