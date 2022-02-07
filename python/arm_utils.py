import numpy as np
import cv2
import time
from general_utils import *

# display_zone provides the length/width/height/depth of cuboid to segment out, in mm

def retrieve_arm_mask(depth_map, bright_map, intrinsics_map, display_zone=[150, 50, 10, 35]):
    
    ##### fixed parameters
    
    # for pixels further than d_threshold, use a fixed brightness threshold of bright_cut[0], else use edge as threshold
    # feels off, probably can rely on edge only... but not a big priority
    bright_cut = [1000, 0.01]
    d_threshold = 250
    
    # 3d dimensions of the marker, length and width (last value is the mimimum width that is acceptable, actually probably redundant?)
    dimensions = [32, 12, 5]
    # margin of error for the above 3d dimensions
    dimensions_margin = 10    
    # 2d threshold (regarding how well the contour fits the best possible rectangle)
    threshold_2d = 0.6    
    
    far_map = (depth_map > d_threshold).astype(int)
    recon_far = (bright_map > bright_cut[0]).astype(int) * far_map
    near_map = (far_map * -1) + 1
    recon_near = (cv2.Sobel(bright_map, cv2.CV_32F, 1, 0) > 1000).astype(np.uint8)
    recon_near = cv2.dilate(recon_near, np.ones((2,2), np.uint8), iterations = 1) * near_map
    recon_full = recon_far + recon_near
    
    contours, _ = cv2.findContours(recon_full.astype(np.uint8), 0, method=cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if len(c) > 20]
    quads = []
    quad_details = []
    cont_preserve = []
    temp1 = np.zeros((512,512))
    
    for cont in contours:
        
        conth = cv2.convexHull(cont)
        rect = cv2.minAreaRect(conth)
        box = cv2.boxPoints(rect)
        
        # 2d area check
        temp1 = temp1 * 0
        cv2.drawContours(temp1,[cont],0,1,-1)
        if np.sum(temp1) / (rect[1][0] * rect[1][1]) < threshold_2d:
            print("2D failed")
            continue
        
        # 3d dimensions check
        try:
            status, details = check3d(depth_map, intrinsics_map, temp1, box, dimensions, dimensions_margin)
        except:
            status = False
        if not status:
            print("3D failed")
            continue
            
        cont_preserve.append(cont)
        quads.append(box)
        quad_details.append(details)
        
    if len(quads) > 1:
        
        ds = np.array([d[-1] for d in quad_details]) # mean depth for each quad
        cont_preserve = [cont_preserve[np.argmin(ds)]] # select closest quad
        quads = [quads[np.argmin(ds)]]
        quad_details = [quad_details[np.argmin(ds)]]
        
    segmap1 = np.zeros((512,512))
    if len(quads) > 0: 

        # can cut processing time here (ps, ps1, ps2 are not needed for algo, for display only)
        # take the remaining contour, if any, and get the 3d boundaries (first output is for visualization, second defines 4 points that characterizes the cuboid)
        #[ps, ps1, ps2], supports_for_3d = get_search_zone(quad_details[0], depth_map, intrinsics_map, display_zone)
        supports_for_3d = get_search_zone(quad_details[0], depth_map, intrinsics_map, display_zone, visual=0)
        
        # search for valid pixels, which give 3d points that are within the cuboid
        segmap1 = search_cuboid(depth_map, intrinsics_map, supports_for_3d) 
        # remove the pixels that are part of the marker (too bright)
        segmap1 = segmap1 - cv2.drawContours(np.zeros((512,512)),[np.expand_dims(quads[0], 1).astype(int)],0,1,-1)
        segmap1[segmap1 < 0] = 0
        # erode away the edges to remove any remnants of the marker. segmap1 is the final output for the arm tracking
        segmap1 = cv2.erode(segmap1, np.ones((3,3), np.uint8), iterations = 1)
        
        # for visualization
        #ps = np.vstack((ps, ps[0,:]))
        #ps1 = np.vstack((ps1, ps1[0,:]))
        #ps2 = np.vstack((ps2, ps2[0,:]))
    
        return np.mean(quads[0], 0), segmap1

    return 0, segmap1


def load_image(file_name):
    return cv2.imread(file_name, -1)  

def check3d(depth, intrinsics, mask, box, dimensions, dimensions_margin):
    
    mask = mask * ((depth > 0) & (depth < 1001)) # updating mask to exclude bad depth
    image_coords = np.array([np.where(mask)[1], np.where(mask)[0]]).transpose() # row, col
    
    if np.sqrt(np.sum((box[0] - box[1]) ** 2)) < np.sqrt(np.sum((box[1] - box[2]) ** 2)):
        short_first = True
    else:
        short_first = False
     
    corner3d = []
    naive3d = []
    corner2d = []
    depth_vals = []

    for b in box:
        closest_pixel = image_coords[np.argmin(np.sum((image_coords - b) ** 2, 1)), :]
        corner2d.append(closest_pixel)        
        #print('debug:', b, closest_pixel, depth[b[1].astype(int), b[0].astype(int)])
        corner3d.append(depth[closest_pixel[1], closest_pixel[0]] * intrinsics[closest_pixel[1], closest_pixel[0]])
        #print(depth.shape)
        #print(depth[b[1].astype(int), b[0].astype(int)] * intrinsics[b[1].astype(int), b[0].astype(int)])
        #print(np.mean((depth[b[1].astype(int)-1:b[1].astype(int)+2, b[0].astype(int)-1:b[0].astype(int)+2] * intrinsics[b[1].astype(int)-1:b[1].astype(int)+2, b[0].astype(int)-1:b[0].astype(int)+2, :]),axis=(0,1)))
        #print(depth[b[1].astype(int)-1:b[1].astype(int)+2, b[0].astype(int)-1:b[0].astype(int)+2])
        #naive3d.append(depth[b[1].astype(int), b[0].astype(int)] * intrinsics[b[1].astype(int), b[0].astype(int)])
        #naive3d.append(np.mean((depth[b[1].astype(int)-1:b[1].astype(int)+2, b[0].astype(int)-1:b[0].astype(int)+2] * intrinsics[b[1].astype(int)-1:b[1].astype(int)+2, b[0].astype(int)-1:b[0].astype(int)+2, :]),axis=(0,1)))        
        naive3d.append(np.median(depth[b[1].astype(int)-1:b[1].astype(int)+2, b[0].astype(int)-1:b[0].astype(int)+2]) * intrinsics[b[1].astype(int), b[0].astype(int)])        
        
        depth_vals.append(depth[closest_pixel[1], closest_pixel[0]])
    corner2d = np.array(corner2d)
    corner3d = np.array(corner3d) # scrutinize this
    naive3d = np.array(naive3d)
    
    # check long length
    if short_first:
        long_d1 = np.sqrt(np.sum((corner3d[1,:] - corner3d[2,:]) ** 2))
        long_d2 = np.sqrt(np.sum((corner3d[3,:] - corner3d[0,:]) ** 2))
    else:
        long_d1 = np.sqrt(np.sum((corner3d[0,:] - corner3d[1,:]) ** 2))
        long_d2 = np.sqrt(np.sum((corner3d[2,:] - corner3d[3,:]) ** 2))
    #print("long: ", long_d1, long_d2)
        
    # check short length
    if not short_first:
        short_d1 = np.sqrt(np.sum((corner3d[1,:] - corner3d[2,:]) ** 2))
        short_d2 = np.sqrt(np.sum((corner3d[3,:] - corner3d[0,:]) ** 2))
    else:
        short_d1 = np.sqrt(np.sum((corner3d[0,:] - corner3d[1,:]) ** 2))
        short_d2 = np.sqrt(np.sum((corner3d[2,:] - corner3d[3,:]) ** 2))
  
    if short_d1 < dimensions[2] or short_d2 < dimensions[2]:
        print('short end too short')
        return False, []
        
    if (np.abs(dimensions[0] - long_d1) > dimensions_margin) or (np.abs(dimensions[0] - long_d2) > dimensions_margin):
        print('long end too different')
        return False, []
        
    if (np.abs(dimensions[1] - short_d1) > dimensions_margin) or (np.abs(dimensions[1] - short_d2) > dimensions_margin):
        print('short end too different')
        return False, []
    
    # todo: check angles for higher robustness
    #
    #
    #
    
    return True, [short_first, naive3d, np.mean(depth_vals)]
    #return True, [short_first, corner3d, np.mean(depth_vals)]


def get_search_zone(details, depth, intrinsics, zone_params, visual=1):
    
    # details - short_order, 3d values, mean depth
    # zone_params - extra projection in length, width, double depth

    if not details[0]: # long order
        details[1] = details[1][[1,2,3,0], :]
    # now first two points will form one short end
    points = details[1]
    
    shortmid1 = np.mean(points[[0,1],:], 0)
    shortmid2 = np.mean(points[[2,3],:], 0)
    longmid1 = np.mean(points[[1,2],:], 0)
    longmid2 = np.mean(points[[3,0],:], 0)
    
    origin = np.mean([shortmid1, shortmid2], 0)
    long_direction = shortmid1 - shortmid2
    long_direction = long_direction / np.sqrt(np.sum(long_direction ** 2))
    short_direction = longmid1 - longmid2
    short_direction = short_direction / np.sqrt(np.sum(short_direction ** 2))
    
    # (v4) 11  12 (v1)
    #      # # #
    #      #   #
    #      #   # 
    #      # # #
    # (v3) 21  22 (v2)
   
    long_end1 = origin + (zone_params[0] * long_direction)
    long_end2 = origin - (zone_params[0] * long_direction)
    corner_11 = long_end1 + (zone_params[1] * short_direction)
    corner_12 = long_end1 - (zone_params[1] * short_direction)
    corner_21 = long_end2 + (zone_params[1] * short_direction)
    corner_22 = long_end2 - (zone_params[1] * short_direction) 
    
    #print(origin)
    #print(long_direction)
    #print(long_end1, long_end2)
    
    if visual:
        norm_vecs = [corner_11 / np.sqrt(np.sum(corner_11 ** 2)), 
                       corner_21 / np.sqrt(np.sum(corner_21 ** 2)), 
                       corner_22 / np.sqrt(np.sum(corner_22 ** 2)), 
                       corner_12 / np.sqrt(np.sum(corner_12 ** 2))]
        norm_vecs = np.array(norm_vecs)
        res = search_intrinsics(norm_vecs, intrinsics, scale_factor=20)
    
    vert1 = np.cross((corner_11 - corner_12), (corner_22 - corner_12))
    vert1 = vert1 / np.sqrt(np.sum(vert1 ** 2))
    vert2 = np.cross((corner_12 - corner_22), (corner_21 - corner_22))
    vert2 = vert2 / np.sqrt(np.sum(vert2 ** 2))
    vert3 = np.cross((corner_22 - corner_21), (corner_11 - corner_21))
    vert3 = vert3 / np.sqrt(np.sum(vert3 ** 2))
    vert4 = np.cross((corner_11 - corner_21), (corner_22 - corner_21))
    vert4 = vert4 / np.sqrt(np.sum(vert4 ** 2))
    vert = np.mean([vert1, vert2, vert3, vert4], 0)
    vert = vert / np.sqrt(np.sum(vert ** 2))
    
    corner_11u = corner_11 + (zone_params[3] * vert)
    if visual:
        corner_12u = corner_12 + (zone_params[3] * vert)
        corner_21u = corner_21 + (zone_params[3] * vert)
        corner_22u = corner_22 + (zone_params[3] * vert)     
    
    corner_11d = corner_11 - (zone_params[2] * vert)
    corner_12d = corner_12 - (zone_params[2] * vert)
    corner_21d = corner_21 - (zone_params[2] * vert)
    if visual:
        corner_22d = corner_22 - (zone_params[2] * vert)  
    
    if visual:
        
        norm_vecs = [corner_11u / np.sqrt(np.sum(corner_11u ** 2)), 
                       corner_21u / np.sqrt(np.sum(corner_21u ** 2)), 
                       corner_22u / np.sqrt(np.sum(corner_22u ** 2)), 
                       corner_12u / np.sqrt(np.sum(corner_12u ** 2))]
        norm_vecs = np.array(norm_vecs)
        resu = search_intrinsics(norm_vecs, intrinsics, scale_factor=20)
        
        norm_vecs = [corner_11d / np.sqrt(np.sum(corner_11d ** 2)), 
                       corner_21d / np.sqrt(np.sum(corner_21d ** 2)), 
                       corner_22d / np.sqrt(np.sum(corner_22d ** 2)), 
                       corner_12d / np.sqrt(np.sum(corner_12d ** 2))]
        norm_vecs = np.array(norm_vecs)
        resd = search_intrinsics(norm_vecs, intrinsics, scale_factor=20)    
        
        return [res, resu, resd], [corner_11d, corner_11u, corner_21d, corner_12d]
    
    return [corner_11d, corner_11u, corner_21d, corner_12d]

def search_cuboid(depth_map, intrinsics_map, supports_for_3d):
    
    # https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
    
    depth = np.reshape(depth_map, (512*512, 1))
    vects = np.reshape(intrinsics_map, (512*512, 3))
    
    x = depth * vects
    
    ps = supports_for_3d[0]
    p1 = supports_for_3d[1]
    p2 = supports_for_3d[2]
    p3 = supports_for_3d[3]
    
    u = np.cross(ps-p2, ps-p3)
    v = np.cross(ps-p1, ps-p3)
    w = np.cross(ps-p1, ps-p2)
    
    ux = np.dot(x, u)
    vx = np.dot(x, v)
    wx = np.dot(x, w)
    
    us = np.dot(ps, u)
    u1 = np.dot(p1, u)
    
    vs = np.dot(ps, v)
    v2 = np.dot(p2, v)
    
    ws = np.dot(ps, w)
    w3 = np.dot(p3, w)
    
    #print(u, v, w)
    #print(depth.shape, vects.shape, x.shape)
    #print(ux.shape)
    
    #print(ux[:10], us, u1)
    #print(vx[:10], vs, v2)
    #print(wx[:10], ws, w3)
    
    valid1 = (ux > np.min([us, u1])) & (ux < np.max([us, u1]))
    valid2 = (vx > np.min([vs, v2])) & (vx < np.max([vs, v2]))
    valid3 = (wx > np.min([ws, w3])) & (wx < np.max([ws, w3]))
    valid_all = valid1 & valid2 & valid3
    
    #print(valid1.shape, np.sum(valid1))
    #print(valid2.shape, np.sum(valid2))
    #print(valid3.shape, np.sum(valid3))
    #print(valid_all.shape, np.sum(valid_all))
    
    return np.reshape(valid_all, (512,512))



def get_vein_mask(file_name):
    image = cv2.imread(file_name,-1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = (image[:,:,0]==255) & ((image[:,:,1]+image[:,:,2])<50)
    mask = np.reshape(mask , (512,512)).astype(int)
    return mask

def get_vein_mask2(file_name):
    image = cv2.imread(file_name,-1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = (image[:,:,0]==255) & ((image[:,:,1]+image[:,:,2])<50)
    mask2 = (image[:,:,1]==255) & ((image[:,:,0]+image[:,:,2])<50)
    mask = np.logical_or(mask, mask2)
    mask = np.reshape(mask , (512,512)).astype(int)
    return mask