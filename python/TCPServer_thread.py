import socket
import struct
import sys
import os
import numpy as np
import cv2
import time
import threading

def tcp_server():
    
    global packet_queue
    serverHost = '' # localhost
    serverPort = 1234
    msg_size = 512*512*4+5
    save_folder = 'data/'
    if ~os.path.isdir(save_folder):
        try:
            os.mkdir(save_folder)
        except:
            pass

    # Create a socket
    sSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind server to port
    try:
        sSock.bind((serverHost, serverPort))
        print('Server bind to port '+str(serverPort))
    except socket.error as msg:
        print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
        return

    sSock.listen(10)
    print('Start listening...')
    sSock.settimeout(3.0)
    while True:
        try:
            conn, addr = sSock.accept() # Blocking, wait for incoming connection
            break
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception:
            continue

    print('Connected with ' + addr[0] + ':' + str(addr[1]))

    experiment_timer = time.time()
    frames_received = 0
    timeout = time.time()
    while True:
        
        if time.time() - timeout > 5:
            print("no valid data received, timing out...")
            break
        
        if time.time() - experiment_timer > 10:
            print("experiment over")
            break
        
        # Receiving from client
        try:
            data = bytes()
            while len(data) < msg_size:
                part = conn.recv(msg_size - len(data))
                if part == '':
                    break
                data += part
            
            if len(data)==0:
                continue
            
            header = data[0:1].decode('utf-8')

            if header == 's':
                packet_queue.append(data)
                frames_received += 1
                timeout = time.time()
                print(frames_received)
                
        except Exception as e:
            print(e)
            break
    
    print("frames received: ", frames_received)
    print('Closing socket...')
    sSock.close()

def data_saver():
    
    global packet_queue
    save_folder = 'data/'
    
    timeout = time.time()
    frames_saved = 0
    while True:
        
        if time.time() - timeout > 5:
            print("queue empty for too long, timing out...")
            break        
        
        if len(packet_queue) > 0:
            single_packet = packet_queue.pop()
            data_length = struct.unpack(">i", single_packet[1:5])[0]
            N = data_length
            depth_img_np = np.frombuffer(data[5:5+N], np.uint16).reshape((512,512))
            ab_img_np = np.frombuffer(data[5+N:5+2*N], np.uint16).reshape((512,512))      
            timestamp = str(int(time.time()))
            cv2.imwrite(save_folder + timestamp+'_depth.tiff', depth_img_np)
            cv2.imwrite(save_folder + timestamp+'_abImage.tiff', ab_img_np) 
            timeout = time.time()
            frames_saved += 1
    
    print("frames saved: ", frames_saved)

def multithreads():

    thread1 = threading.Thread(target=tcp_server(), args=())
    thread2 = threading.Thread(target=data_saver(), args=())
    thread1.start()
    thread2.start()

if __name__ == "__main__":
    
    packet_queue = []
    multithreads()
    
