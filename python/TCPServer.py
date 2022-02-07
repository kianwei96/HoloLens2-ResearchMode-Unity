import socket
import struct
import sys
import os
import numpy as np
import cv2
import time
import open3d as o3d
import pickle as pkl
import time
import datetime

def tcp_server():
    
    write = True
    serverHost = '' # localhost
    serverPort = 9090

    save_folder = 'data/'
    sub_folder = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '/'
    if ~os.path.isdir(save_folder):
        try:
            os.mkdir(save_folder)
        except:
            pass
    os.mkdir(save_folder + sub_folder)
    save_folder = save_folder + sub_folder
    os.mkdir(save_folder + "depth/")
    os.mkdir(save_folder + "left/")
    os.mkdir(save_folder + "right/")

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
    depth_stamper = []
    vlc_stamper = []
    time_start = time.time()
    disp1 = np.zeros((512, 512*2))
    disp2 = np.zeros((640, 480*2))
    dispd1 = np.zeros((512,512))
    LF_img_np = np.zeros((480,640))
    total_count = 0

    while True:
        # Receiving from client
        try:
            #data = conn.recv((512*512*4)+5) = # 512*512*4+100 original
            
            ##################### receiving data #####################
            msg_sizes = np.array([["d", (512*512*4)+5+4+8],
                                  ["v", (480*640*2)+5+4+16],
                                  ["l", (480*640)+5+8]])
            correct_size = np.min(msg_sizes[:,1].astype(int))
            determined = False
            data = bytes()
            while len(data) < correct_size:
                part = conn.recv(correct_size - len(data))
                if part == '':
                    break
                data += part
                if not determined:
                    header = data[0:1].decode('utf-8')
                    try:
                        correct_size = msg_sizes[np.where(msg_sizes[:,0].astype(str) == header)[0][0], 1].astype(int)
                        if header == "d" or header == "v":
                            correct_size += struct.unpack(">i", data[1:5])[0]
                    except Exception as e:
                        print('invalid header: ', header)
                        print('error: ', e)
                        data = bytes()
                        correct_size = np.min(msg_sizes[:,1].astype(int))
                        determined = False
                        data = bytes()
                        break                        
                    determined = True            
            ##################### received data #####################
            
            if len(data)==0:
                continue
            #header = data[0:1].decode('utf-8')
            #print('--------------------------\nHeader: ' + header)
            #print(len(data))

            if header == 'd':
                # save depth sensor images
                tsize = struct.unpack(">i", data[1:5])[0]
                data_length = struct.unpack(">i", data[5:9])[0]
                N = data_length
                depth_img_np = np.frombuffer(data[9:9+N], np.uint16).reshape((512,512))
                ab_img_np = np.frombuffer(data[9+N:9+2*N], np.uint16).reshape((512,512))
                timestamp = int.from_bytes(data[9+2*N:9+2*N+8], byteorder='big')
                transform = data[9+2*N+8:9+2*N+8+tsize].decode('utf-8')
                print(total_count)
                total_count += 1
                if write:
                    cv2.imwrite(save_folder + "depth/" + str(timestamp) + '_depth.pgm', depth_img_np)
                    cv2.imwrite(save_folder + "depth/" + str(timestamp) + '_abImage.pgm', ab_img_np)                  
                    with open(save_folder + 'depthextrinsics.txt','a') as myfile:
                        myfile.write(str(timestamp) + "\n")
                        myfile.write(transform)                
                print('Depth image with ts ' + str(timestamp) + ' is saved')
                #print(transform)
                #print('\n')
                
                # display/debug
                depth_stamper.append(str(timestamp))
                dispab = ab_img_np.copy().astype(float)
                dispab[dispab > 1000] = 1000
                dispab = dispab/np.max([1000, np.max(dispab)])
                dispd = depth_img_np.copy().astype(float)
                dispd[dispd > 1000] = 0
                dispd = dispd/np.max(dispd)                  
                disp1 = cv2.hconcat((dispab, dispd)) 
                cv2.imshow('window1', disp1)   
                dispd1 = np.vstack((np.zeros((640-512, 512)), dispd))
                #cv2.imwrite('data/' + str(timestamp) + '_depth.png', dispd1*255)
                
            if header == 'v':
                # save spatial camera images
                tsize = struct.unpack(">i", data[1:5])[0]
                data_length = struct.unpack(">i", data[5:9])[0]
                ts_left, ts_right = struct.unpack(">qq", data[9:25])

                N = int(data_length/2)
                LF_img_np = np.frombuffer(data[25:25+N], np.uint8).reshape((480,640)) # 1x480x640 bytes
                RF_img_np = np.frombuffer(data[25+N:25+2*N], np.uint8).reshape((480,640))
                transforms = data[25+2*N:25+2*N+tsize].decode('utf-8')
                left_transform = "\n".join(transforms.split("\n")[:4])
                right_transform = "\n".join(transforms.split("\n")[4:])
                print(total_count)
                total_count += 1  
                if write:
                    cv2.imwrite(save_folder + "left/" + str(ts_left) + '_left.pgm', LF_img_np)
                    cv2.imwrite(save_folder + "right/" + str(ts_right) + '_right.pgm', RF_img_np)                 
                    with open(save_folder + 'leftextrinsics.txt','a') as myfile:
                        myfile.write(str(ts_left) + "\n")
                        myfile.write(left_transform)  
                        myfile.write('\n')
                    with open(save_folder + 'rightextrinsics.txt','a') as myfile:
                        myfile.write(str(ts_right) + "\n")
                        myfile.write(right_transform)                                    
                print('Stereo images with ts %d and %d is saved' % (ts_left, ts_right))
                #print(transforms)
                #print('\n')                
                
                # display/debug
                vlc_stamper.append([str(ts_left), str(ts_right)])
                disp2 = cv2.hconcat((np.flip(np.transpose(LF_img_np), axis=1), np.flip(np.transpose(RF_img_np), axis=0)))
                cv2.imshow('window2', disp2)   
                #cv2.imwrite('data/' + str(ts_left) + '_vlc.png', np.flip(np.transpose(LF_img_np), axis=1))
                
            if header == 'l':
                # save lf camera images
                data_length = struct.unpack(">i", data[1:5])[0]
                ts_left = struct.unpack(">q", data[5:13])

                N = data_length
                print(np.frombuffer(data[13:13+N], np.uint8).shape)
                LF_img_np = np.frombuffer(data[13:13+N], np.uint8).reshape((480,640)) # 1x480x640 bytes
                #cv2.imwrite(save_folder + str(ts_left)+'_LF.tiff', LF_img_np)
                print('Left image with ts %d and is saved' % (ts_left))
                vlc_stamper.append(ts_left)
                disp2 = np.flip(np.transpose(LF_img_np), axis=1)
                cv2.imshow('window2', disp2)                  
            if header == 'p':
                # save point cloud
                N_pointcloud = struct.unpack(">i", data[1:5])[0]
                print("Length of point cloud:" + str(N_pointcloud))
                pointcloud_np = np.frombuffer(data[5:5+N_pointcloud*3*4], np.float32).reshape((-1,3))
                
                timestamp = str(int(time.time()))
                temp_filename_pc = timestamp + '_pc.ply'
                print(pointcloud_np.shape)
                o3d_pc = o3d.geometry.PointCloud()
                o3d_pc.points = o3d.utility.Vector3dVector(pointcloud_np.astype(np.float64))
                o3d.io.write_point_cloud(save_folder + temp_filename_pc, o3d_pc, write_ascii=True)
                print('Saved  image to ' + temp_filename_pc)
            
                
            key = cv2.waitKey(1)
            #print(len(depth_stamper))
            if key == ord('q'):
                break     

        except Exception as e:
            print('exception: ', e)
            break
    
    print('Closing socket...')
    sSock.close()
    cv2.destroyAllWindows()
    
    print('Statistics: ')
    print('Time listening: ', time.time() - time_start)
    print('Depth frames: ', len(depth_stamper))
    print('VLC frames: ', len(vlc_stamper))
    print('Exiting...')


if __name__ == "__main__":
    tcp_server()
