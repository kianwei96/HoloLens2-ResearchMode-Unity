import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
import torch
import itertools
from general_utils import *

#### ResearchMode Timestamp Matching

def find_matching_transform(rightr2w, right_time):
    index = np.argmin(np.abs(np.uint64(rightr2w[:,0]) - right_time))
    if np.abs(np.uint64(rightr2w[index,0]) - right_time) / 1e4 < 2:
        return True, np.reshape(np.float64(rightr2w[index, 1:]), (4, 4))
    return False, 0

#### Old IR Marker Searching (Depreciated)

def find_triplets(cont1, depth, intrinsics): # depth in m
    
    if len(cont1) < 3:
        print('failed at stage 0')
        return False, 0, 0
    
    image_loc = [np.mean(c.squeeze(), 0).astype(int) for c in cont1]
    max_dist = [np.max(np.sum((cont1[i] - image_loc[i])**2, 1)) for i in range(len(cont1))]
    
    local_loc = [depth[i[1]][i[0]]*intrinsics[i[1]][i[0]] for i in image_loc] # intrinsics need to flip?
    #px_size =  [np.sum(cv2.drawContours(np.zeros((512,512)), [c], -1, 1, -1)) for c in cont1]
    px_size = [len(c) for c in cont1] # approximate area with perimeter length
    
    pd = pairwise_distances(local_loc)
    sd = pairwise_distances(np.array(px_size).reshape(-1, 1))
    md = pairwise_distances(np.array(max_dist).reshape(-1, 1))
    px_size = np.array(px_size)
    max_dist = np.array(max_dist)
    px_min = np.ones((len(px_size),len(px_size)))
    px_max = np.ones((len(px_size),len(px_size)))
    dist_min = np.ones((len(max_dist), len(max_dist)))
    for i in range(len(px_size)):
        for j in range(i+1, len(px_size)):
            px_min[i,j] = np.min(px_size[[i,j]])
            px_min[j,i] = px_min[i,j] 
            px_max[i,j] = np.max(px_size[[i,j]])
            px_max[j,i] = px_max[i,j]             
            dist_min[i,j] = np.min(max_dist[[i,j]])
            dist_min[j,i] = dist_min[i,j]             
    sdp = 100 * (sd / px_min)
    mdp = 100 * (md / dist_min)
    
    ret, cd_idx = adj_mat_splitter(pd, px_max, sdp, mdp) 
    
    if not ret:
        print('failed to find triplet candidate')
        return False, 0, 0
    
    lengths = [pd[r[0], r[1]] for r in cd_idx]
    
    temp = np.delete(cd_idx, np.argmax(lengths), axis=0)
    unique, counts = np.unique(temp.flatten(), return_counts=True)
    anchor = int(unique[np.argmax(counts)])
    
    temp_length = np.delete(lengths, np.argmax(lengths))
    short_candidates = temp[np.argmin(temp_length)]
    short_end = int(short_candidates[short_candidates != anchor])
        
    long_candidates = temp[np.argmax(temp_length)]
    long_end = int(long_candidates[long_candidates != anchor])
    
    #print(time.time() - time_start)
    
    return True, np.array(image_loc)[[anchor, short_end, long_end]], np.array(local_loc)[[anchor, short_end, long_end]]

def adj_mat_splitter(pair_dist, px_max, size_perc_inc, spread_perc_inc):

    exemption = 36
    pd1 = (pair_dist > 1.5/100) & (pair_dist < 7.0/100) # v5 was 3.5 and 8.5, maybe 6.5 can be considered (5max IRL)
    #sd1 = (size_perc_inc < 75) #| (px_max < exemption)
    #md1 = (spread_perc_inc < 500) #| (px_max < exemption)
    sd1 = (size_perc_inc < (75 * (1+(px_max < exemption))))
    md1 = (spread_perc_inc < (500  * (1+(px_max < exemption))))
    
    #print(size_perc_inc.astype(int))
    #print(spread_perc_inc.astype(int))
    #print(md1.astype(int))
    #print(sd1.astype(int))
    
    matrix = (pd1 & sd1 & md1).astype(int)    
    
    #print(matrix)
    
    found = []
    
    temp = np.sum(matrix, 1) == 2
    doublets = matrix[temp]
    candidates = np.where(temp)[0]

    checked = np.zeros(len(candidates))
    for candidx, candidate in enumerate(candidates):
        if checked[candidx] == 1:
            continue
        try:
            partners = np.where(doublets[candidx])[0]
            row1 = np.where(candidates == partners[0])[0]
            part1 = np.where(doublets[row1[0]])[0]
            row2 = np.where(candidates == partners[1])[0]
            part2 = np.where(doublets[row2[0]])[0]
            uniques = np.unique([partners, part1, part2])
        except:
            checked[candidx] = 1
            continue
        if len(uniques) == 3:
            found.append(uniques)
        checked[[candidx, row1[0], row2[0]]] = 1
    
    if len(found) == 1:    
        three_pairs = np.array([[found[0][0], found[0][1]], [found[0][0], found[0][2]], [found[0][1], found[0][2]]])
        return True, three_pairs
    
    if len(found) == 0:
        return False, 0
    
    if len(found) > 1:
        #print('debug')
        scores = []
        all_pairs = []
        for idx in range(len(found)):
            three_pairs = np.array([[found[idx][0], found[idx][1]], [found[idx][0], found[idx][2]], [found[idx][1], found[idx][2]]])
            distances = [pair_dist[r[0], r[1]] for r in three_pairs]
            hypo = distances.pop(np.argmax(distances))
            recon_length = np.sqrt(np.sum(np.power(distances, 2)))
            score = 100 * np.abs(recon_length - hypo) / np.min([recon_length, hypo])
            scores.append(score)
            #print(score)
            all_pairs.append(three_pairs)
        return True, all_pairs[np.argmin(scores)]
    
#### New IR Marker Searching

# rationale:
# - old method relies on well isolated triplets
# - old method does not exploit plane characteristics
# - basically, find_triplets method does not allow for flexible holding/placing on surfaces
# - it currenlty also relies on poor/finicky constraints like 2d area similarity
# for consideration - surface normals, edge characteristics

def triangle_check(anc, short, long, short_d=0.0245, long_d=0.039, margin=0.0025, verbose=0):
    if verbose:
        print('func: trangle_check')
        print(np.abs(np.sqrt(np.sum((anc - short) ** 2)) - short_d))
        
    if np.abs(np.sqrt(np.sum((anc - short) ** 2)) - short_d) > margin:
        return False

    if verbose:
        print(np.abs(np.sqrt(np.sum((anc - long) ** 2)) - long_d))
        
    if np.abs(np.sqrt(np.sum((anc - long) ** 2)) - long_d) > margin:
        return False

    short_vec = short - anc
    long_vec = long - anc
  
    if np.abs(np.sqrt(np.sum(short_d**2 + long_d**2)) - np.sqrt(np.sum((long-short)**2))) > margin * 2.5:
        return False

    return True

def edge_check(anc3, short3, long3, intrinsics, binary_ir, distances=[0.015,0.038], th=0.65):
    
    # not in use (finicky thresholding)
    
    #intrinsics = intrinsics.swapaxes(0,1)
    #print(intrinsics.shape)
    
    #print(anc3, short3, long3)
    d_short = anc3 - short3 # towards anchor
    d_short = d_short / np.sqrt(np.sum(d_short**2))
    #print(d_short)
    edge1_pts = np.array([anc3, long3] + (d_short * distances[0]))
    edge2_pts = np.array([anc3, long3] - (d_short * distances[1]))
    
    edge1_norm = [edge1_pts[0] / np.sqrt(np.sum(edge1_pts[0] ** 2)), edge1_pts[1] / np.sqrt(np.sum(edge1_pts[1] ** 2))]
    edge2_norm = [edge2_pts[0] / np.sqrt(np.sum(edge2_pts[0] ** 2)), edge2_pts[1] / np.sqrt(np.sum(edge2_pts[1] ** 2))]
    

    norms = np.vstack([edge1_norm, edge2_norm])
    px_edges = search_intrinsics(norms, intrinsics, scale_factor=16)
    
    #print(px_edges)
    #return [anc3, short3, long3, edge1_pts, edge2_pts, edge1_norm, edge2_norm]
    
    canvas0 = cv2.line(np.zeros((512,512)), (px_edges[0,0], px_edges[0,1]), (px_edges[1,0], px_edges[1,1]), 1)
    area0 = np.sum(canvas0)
    #canvas0 = cv2.dilate(canvas0, np.ones((1,1), np.uint8))
    edge_overlap = np.sum(canvas0.transpose() * binary_ir) / area0
    canvas1 = cv2.line(np.zeros((512,512)), (px_edges[2,0], px_edges[2,1]), (px_edges[3,0], px_edges[3,1]), 1)
    area1 = np.sum(canvas1)
    #canvas1 = cv2.dilate(canvas1, np.ones((1,1), np.uint8))
    edge_overlap2 = np.sum(canvas1.transpose() * binary_ir) / area1
    print('edge overlap: ', edge_overlap, edge_overlap2)
    #return (canvas0 + canvas1)
    if np.max(edge_overlap) > th and np.mean(edge_overlap) > 0.5:
        return True
    
    return False

def surface_check(anc3, short3, long3, intrinsics_map, depth_map, tol=0.025, verbose=0):
    
    d_short = short3 - anc3
    d_long = long3 - anc3
    
    dims = [3, 4]
    depths = []
    norm_vecs = []
    for h_interval in range(dims[0]):
        for v_interval in range(dims[1]):
            pt3 = anc3 + (h_interval/(dims[0]-1))*d_short + (v_interval/(dims[1]-1))*d_long
            #print('plane part: ', pt3)
            depths.append(np.sqrt(np.sum(pt3**2)))
            norm_vecs.append(pt3/np.sqrt(np.sum(pt3**2)))
    pxs = search_intrinsics(norm_vecs, intrinsics_map, scale_factor=16)
        
    diff = [np.abs((depths[didx] - depth_map[pxs[didx,0], pxs[didx,1]])/depths[didx]) for didx in range(len(depths))]
    
    if verbose:
        print('func: surface_check')
        print(np.median([np.abs((depths[didx] - depth_map[pxs[didx,0], pxs[didx,1]])) for didx in range(len(depths))]))
        print(np.mean(depths), np.mean(diff), np.percentile(diff, 75), tol)
        
    if (np.mean(diff) < tol) and (np.percentile(diff, 75) < tol*2):
        return True
            
    return False    
    
def candidate_search(cont1, depth_map, intrinsics_map, binary_ir, markers_max=2, verbose=0, triangle_tol=0.0025, sfc_tol=0.025): # depth in metres
    
    binary_ir = cv2.dilate(binary_ir.astype(np.uint8), np.ones((4,4), np.uint8), iterations=3)
    cont1 = [c for c in cont1 if len(c) > 2]  
    
    try:
        # get contours with 3d positions (c0)
        if len(depth_map.shape) > 2:
            depth_map = depth_map[:,:,0]
        contour_3d = [np.array([depth_map[i[1]][i[0]]*intrinsics_map[i[1]][i[0]] for i in c.squeeze() if depth_map[i[1],i[0]] > 0.001]) for c in cont1]
        contour_3d = [c for c in contour_3d if len(c) > 1]
        contour_3d_mid = np.array([np.mean(c, 0) for c in contour_3d])
        
        # filter max distance to centre <1cm into candidate markers (c1)
        # actual dimensions - 5mm diameter (marker)
        threshold_shape = [0.001, 0.01] # minmax on max dist to mid
        max_contour_dist = np.array([np.sqrt(np.max(np.sum((contour_3d[ci] - contour_3d_mid[ci])**2,1))) for ci in range(len(contour_3d_mid))])
        marker_candidate = np.array(contour_3d)[np.where((max_contour_dist < threshold_shape[1]) & (max_contour_dist > threshold_shape[0]))[0]]
        marker_candidate_mid = contour_3d_mid[np.where((max_contour_dist < threshold_shape[1]) & (max_contour_dist > threshold_shape[0]))[0],:]
        marker_candidate_pixels = [cont1[ci] for ci in range(len(contour_3d)) if ((max_contour_dist[ci] < threshold_shape[1])  & (max_contour_dist[ci] > threshold_shape[0]))]
    except: # actually i havent tracked down the issue here, but rare
        return False, 0, 0

    if verbose:
        print('func: candidate_search')
        print('candidate count: ', len(marker_candidate))
        
    if len(marker_candidate) == 0:
        return False, 0, 0
    
    # filter (c1) for contours (c2) with at least 2 neighbors in (5cm-ish range)
    # actual distance - 24.5mm (short), 39.0mm (long), 46.1mm (hypo)
    threshold_d = [0.005, 0.055] # minmax on dist between markers
    pd = pairwise_distances(marker_candidate_mid)
    pd = (pd > threshold_d[0]) & (pd < threshold_d[1])
    anchor_candidates = [[ridx, np.where(pd[ridx])[0]] for ridx in range(len(pd)) if np.sum(pd[ridx]) > 1]
    
    # for each (c2), find 2 best neighbors that fit the length/angle requirements. only keep contours (c3) that are good anchors
    # actual distance - 24.5mm (short), 39.0mm (long)
    filtered = []
    for ac in anchor_candidates:
        for (short, long) in itertools.permutations(ac[1], 2):
            if verbose:
                print(ac[0], short, long)
            valid = triangle_check(marker_candidate_mid[ac[0]], marker_candidate_mid[short], marker_candidate_mid[long], verbose=verbose, margin=triangle_tol)
            if valid:
                filtered.append([ac[0], short, long])
    if verbose:
        print(filtered)
    #return marker_candidate_pixels[0], marker_candidate_pixels[1], marker_candidate_pixels[2]
    #return marker_candidate_pixels[filtered[1][0]], marker_candidate_pixels[filtered[1][1]], marker_candidate_pixels[filtered[1][2]]

    # in (c3) triplets, test for planar constraint (6/9 pt grid), keep maximum of two as (c4) 
    filtered2 = []
    for i in range(len(filtered)):
        if verbose:
            print(filtered[i])
        status = surface_check(marker_candidate_mid[filtered[i][0]], marker_candidate_mid[filtered[i][1]], marker_candidate_mid[filtered[i][2]], intrinsics_map, depth_map, verbose=verbose, tol=sfc_tol)
        #return status
        if status:
            filtered2.append(filtered[i])
    if verbose:
        print(filtered2)
    
    # in (c4), assign based off gaze/accumulated positions    
    
    if len(filtered2) > markers_max:
        return False, 0, len(filtered2)
    
    imlocs = []
    loclocs = []
    for m in range(len(filtered2)):
        imloc = []
        locloc = []
        for i in range(3):
            imloc.append(np.mean(marker_candidate_pixels[filtered2[m][i]].squeeze(), 0).astype(int))
            locloc.append(np.mean(marker_candidate[filtered2[m][i]], 0))
        imloc = np.array(imloc)
        locloc = np.array(locloc)
        imlocs.append(imloc)
        loclocs.append(locloc)

    return True, np.array(imlocs), np.array(loclocs)

#### 2-Phase Grid/LUT Matching

def search_intrinsics(norm, intrinsics, scale_factor=20):
    
    # subsample original grid
    iss = intrinsics[::scale_factor, ::scale_factor, :]
    issf = np.reshape(iss, (-1,3))   

    # search in subsampled grid
    mins = cdist(norm, issf)
    mins = np.argmin(mins, axis=1)
    pixels = np.unravel_index(mins, iss.shape[:-1])
    pixels = np.array(pixels).transpose()
    pixels = pixels * scale_factor
    
    # create windowed focus area
    row_start = np.max([np.min(pixels[:,0] - (scale_factor + 5)), 0])
    row_end = np.min([np.max(pixels[:,0] + (scale_factor + 5)), intrinsics.shape[0] - 1])
    col_start = np.max([np.min(pixels[:,1] - (scale_factor + 5)), 0])
    col_end = np.min([np.max(pixels[:,1] + (scale_factor + 5)), intrinsics.shape[1] - 1])
    
    # search in focus area
    i_segment = intrinsics[row_start:row_end, col_start:col_end, :]
    i_segmentf = np.reshape(i_segment, (-1,3))
    mins = cdist(norm, i_segmentf)
    #print(np.min(mins, 1))
    mins = np.argmin(mins, axis=1)
    pixels = np.unravel_index(mins, i_segment.shape[:-1])
    pixels = np.array(pixels).transpose()
    pixels += [row_start, col_start] # place back in big grid
    
    return pixels

# Projection from IR to Stereo, and Cropping

def static_cropper(image, loc, px=30):
    padded = cv2.copyMakeBorder(image.copy(),px,px,px,px,cv2.BORDER_CONSTANT,0)
    return padded[loc[1]:loc[1]+2*px, loc[0]:loc[0]+2*px]

def reject_border_projections(pixels, dims=[480, 640], border=1):
    if np.sum((pixels[:,0] < (border+1)) | (pixels[:,0] > dims[0]-(border+1))) > 0:
        return True
    if np.sum((pixels[:,1] < (border+1)) | (pixels[:,1] > dims[1]-(border+1))) > 0:
        return True
    return False

def project_onto_stereo(locloc, d2right, d2left, right_intrinsics, left_intrinsics, border=3, strict=1):
    
    depth_homo = np.vstack([locloc.transpose(), np.ones(locloc.shape[0])]).transpose() # correct orientation?

    right_homo = d2right.dot(depth_homo.transpose()).transpose()
    right_norm = (right_homo.transpose() / np.sqrt(np.sum(right_homo[:,:-1] ** 2, 1))).transpose()[:,:-1]

    left_homo = d2left.dot(depth_homo.transpose()).transpose()
    left_norm = (left_homo.transpose() / np.sqrt(np.sum(left_homo[:,:-1] ** 2, 1))).transpose()[:,:-1]        

    right_pixels = search_intrinsics(right_norm, right_intrinsics)
    left_pixels = search_intrinsics(left_norm, left_intrinsics)

    valid_dims = [right_intrinsics.shape[0], right_intrinsics.shape[1]]
    if strict:
        if reject_border_projections(right_pixels, valid_dims, border) or reject_border_projections(left_pixels, valid_dims, border):
            print('border detected, invalid')
            return False, 0, 0   
        
    return True, right_pixels, left_pixels

#### Segmentation with EllipSegNet, and Ellipse Fitting

def eval_contours(masks, area_t=0.5):
    centers = []
    area_e = []
    for mask in masks:
        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        idx = 0
        if len(contours) > 1:
            areas = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                areas.append(area)
            idx = np.argmax(areas).item()
        
        try:
            rbox = cv2.fitEllipse(contours[idx])
        except:
            return False, 0
        centers.append([rbox[0][0],rbox[0][1]])
        area_e.append(np.pi * rbox[1][0] * rbox[1][1])
    #print(area_e)
    for i in range(3):
        min_area = np.min([area_e[i], area_e[i+3]])
        if np.abs(area_e[i] - area_e[i+3]) / min_area > area_t: # cannot be more than 40% difference in ellipse area between L/R
            print('failed area test')
            return False, 0
    
    return True, np.array(centers)

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        return self.double_conv(x)

class EllipSegNet(torch.nn.Module):
    def __init__(self, init_f, num_outputs):
        super(EllipSegNet, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.inc = DoubleConv(1, init_f)
        self.down1 = DoubleConv(init_f, 2*init_f)
        self.down2 = DoubleConv(2*init_f, 4*init_f)
        self.down3 = DoubleConv(4*init_f, 4*init_f)
        self.up1 = DoubleConv(2*4*init_f, 2*init_f, 4*init_f)
        self.up2 = DoubleConv(2*2*init_f, init_f, 2*init_f)
        self.up3 = DoubleConv(2*init_f, init_f)
        self.outc = torch.nn.Conv2d(init_f, num_outputs, kernel_size=1)

    def forward(self, x):
        #x = self.upsample(x) # kw added
        x1 = self.inc(x) #(120,120)
        x2 = self.down1(self.pool(x1)) #(60,60)
        x3 = self.down2(self.pool(x2)) #(30,30)
        x4 = self.down3(self.pool(x3)) #(15,15)
        
        x = torch.cat([self.upsample(x4), x3], 1) #(15*2,15*2), (30,30)
        x = self.up1(x)
        
        x = torch.cat([self.upsample(x), x2], 1) #(30*2,30*2), (60,60)
        x = self.up2(x)
        
        x = torch.cat([self.upsample(x), x1], 1) #(60*2,60*2), (120,120)
        x = self.up3(x)
        
        x = self.outc(x) #(120,120)
        return x
    
#### Solving Intersection of Stereo Pairs

def recover_rays(c_left, c_right, left_intrinsics, right_intrinsics, left2w, right2w):

    #left2w = r2w_left.dot(np.linalg.inv(r2left))
    left_origin = left2w.dot([0,0,0,1])[:-1]

    #right2w = r2w.dot(np.linalg.inv(r2right))
    right_origin = right2w.dot([0,0,0,1])[:-1]

    left_vec = []
    right_vec = []

    for pair_idx in range(3):

        components = interpolate_grid(c_left[pair_idx], left_intrinsics)
        left_vec.append(left2w.dot(np.append(components, 1))[:-1])
        
        components = interpolate_grid(c_right[pair_idx], right_intrinsics)
        right_vec.append(right2w.dot(np.append(components, 1))[:-1])        
        
#         bounds = np.array([np.floor(c_left[pair_idx]), np.ceil(c_left[pair_idx])]).transpose()
#         components = [ [bounds[0][0], bounds[1][0]], [bounds[0][0], bounds[1][1]], [bounds[0][1], bounds[1][0]], [bounds[0][1], bounds[1][1]] ]
#         weights = cdist(components, np.reshape(c_left[pair_idx], (-1,2)))
#         weights = 1 / (weights+1e-7)
#         weights = weights / np.sum(weights)
        
#         components = np.array(components).astype(int)
#         components = left_intrinsics[components[:,0], components[:,1],:]
#         components = np.sum(components * weights, 0)
#         left_vec.append(left2w.dot(np.append(components, 1))[:-1])

#         bounds = np.array([np.floor(c_right[pair_idx]), np.ceil(c_right[pair_idx])]).transpose()
#         components = [ [bounds[0][0], bounds[1][0]], [bounds[0][0], bounds[1][1]], [bounds[0][1], bounds[1][0]], [bounds[0][1], bounds[1][1]] ]
#         weights = cdist(components, np.reshape(c_right[pair_idx], (-1,2)))
#         weights = 1 / (weights+1e-7)
#         weights = weights / np.sum(weights)
#         components = np.array(components).astype(int)
#         components = right_intrinsics[components[:,0], components[:,1],:]
#         components = np.sum(components * weights, 0)
#         right_vec.append(right2w.dot(np.append(components, 1))[:-1])


    left_vec = np.array(left_vec)
    right_vec = np.array(right_vec)
    
    return left_origin, right_origin, left_vec, right_vec    

def intersect_solver(oleft, oright, leftpts, rightpts, th=0.001):

    # https://www.gamedev.net/forums/topic/520233-closest-point-on-a-line-to-another-line-in-3d/
    # (La . La)s - (La . Lb)t = La . (PB1 - PA1), and
    # (La . Lb)s - (Lb . Lb)t = Lb . (PB1 - PA1)    
    
    end_left = []
    end_right = []
    
    for pair_idx in range(3):

        d_left = leftpts[pair_idx] - oleft
        d_right = rightpts[pair_idx] - oright

        p_left = leftpts[pair_idx]
        p_right = rightpts[pair_idx]

        premult = np.array([[d_left.dot(d_left), -d_left.dot(d_right)],
                            [d_left.dot(d_right), -d_right.dot(d_right)]])

        rhs = np.array([d_left.dot(oright-oleft), d_right.dot(oright-oleft)])

        coeff = np.linalg.solve(premult, rhs)
        end_left.append(oleft + coeff[0]*d_left)
        end_right.append(oright + coeff[1]*d_right)
        
    end_left = np.array(end_left)
    end_right = np.array(end_right)
    
    if np.sum(np.abs(end_left - end_right) > th):
        print('bad stereo')
        #print(np.abs(end_left - end_right))
        return False, 0, 0
        
    return True, end_left, end_right

def interpolate_grid(subpixel_coords, intrinsics_grid):
    
    bounds = np.array([np.floor(subpixel_coords), np.floor(subpixel_coords)+1]).transpose().astype(int)

    o1 = intrinsics_grid[bounds[0,0], bounds[1,0]]
    d1 = subpixel_coords[0] - bounds[0,0]
    v1 = intrinsics_grid[bounds[0,1], bounds[1,0]] - o1
    out1 = o1 + d1 * v1
    
    o1 = intrinsics_grid[bounds[0,0], bounds[1,1]]
    v1 = intrinsics_grid[bounds[0,1], bounds[1,1]] - o1  
    out2 = o1 + d1 * v1
    
    v2 = out2 - out1
    d2 = subpixel_coords[1] - bounds[1,0]
    bilinear_out = out1 + d2 * v2    
    
    return bilinear_out