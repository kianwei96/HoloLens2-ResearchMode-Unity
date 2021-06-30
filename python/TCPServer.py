import socket
import struct
import sys
import os
import numpy as np
import cv2
import time
import threading
import csv

def tcp_server():
    serverHost = '' # localhost
    serverPort = 1234
    save_folder = 'data/'

    if ~os.path.isdir(save_folder):
        try:
            os.mkdir(save_folder)
        except:
            pass
        
    if ~os.path.isdir(save_folder + "depth/"):
        try:
            os.mkdir(save_folder + "depth/")
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
    ts = []
    frames_received = 0
    temp_received = 0
    while True:
        # Receiving from client
        
        if time.time() - experiment_timer > 60:
            print("experiment over")
            break        
        
        try:
            
            msg_size_d = (512*512*4)+5+8
            msg_size_t = 1+4+8
            correct_size = np.min([msg_size_d, msg_size_t])
            determined = False
            data = bytes()
            while len(data) < correct_size:
                part = conn.recv(correct_size - len(data))
                if part == '':
                    break
                data += part
                if not determined:
                    header = data[0:1].decode('utf-8')
                    if header == "d":
                        correct_size = msg_size_d
                    elif header == "t":
                        correct_size = msg_size_t
                    determined = True
            
            #data = conn.recv(#512*512*4+100) #+100/+5
            if len(data)==0:
                continue
            #print("data packet received: ", len(data))
            header = data[0:1].decode('utf-8')
            #print('--------------------------\nHeader: ' + header)
            
        
            if header == 'd':
                # get the init transform
                data_length = struct.unpack(">i", data[1:5])[0]
                N = data_length
                depth_img_np = np.frombuffer(data[5:5+N], np.uint16).reshape((512,512))
                ab_img_np = np.frombuffer(data[5+N:5+2*N], np.uint16).reshape((512,512))
                #timestamp = str(int(time.time()))
                timestamp = int.from_bytes(data[5+2*N:5+8+2*N], byteorder='big')
                ts.append(timestamp)   
                cv2.imwrite(save_folder + "depth/" + str(timestamp) + str(frames_received) + '_depth.pgm', depth_img_np)
                cv2.imwrite(save_folder + "depth/" + str(timestamp) + str(frames_received) + '_abImage.pgm', ab_img_np)
                #print('Image with ts ' + timestamp + ' is saved')
                frames_received += 1
                print(frames_received)
                
            if header == 't':
                temperature = int.from_bytes(data[1:1+4], byteorder='big')
                timestamp = int.from_bytes(data[1+4:1+4+8], byteorder='big')
                with open(save_folder + 'temperature.csv','a',newline='') as myfile:
                    wrtr = csv.writer(myfile, delimiter=',')
                    wrtr.writerow([timestamp, temperature])
                    myfile.flush()
                temp_received += 1
                
        except Exception as e:
            print(e)
            break
    
    print("frames received: ", frames_received)
    print("temperature values received: ", temp_received)
    print('Closing socket...')
    sSock.close()
    print('Exiting')


if __name__ == "__main__":
    tcp_server()
