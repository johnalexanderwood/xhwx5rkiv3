import cv2
from functools import reduce
import numpy as np
import joblib
from scipy.fftpack import dct
from sklearn.neural_network import MLPClassifier
import time


class RockLensBackend:
    def __init__(self, config):
        self.config = config
        #self.images = images
        #self.widgets = widgets
        #self.id = id # This is the plugin id that this backend belongs to

        self.preprocessing_complete = False

        # Needed for model and learning
        self.img_test_hip = None  # Setup during preprocessing
        self.predictions = None  # prediction by class number, for metrics
        self.x_inputs = np.zeros((1, 256), dtype=np.float64)  # Size manually calculated based on feature setup
        self.y_labels = np.zeros((1,), dtype=np.int64)
        self.z_coords = np.zeros((1, 3), dtype=np.int64)
        self.tile_index = None
        self.tile_grid = None
        self.model = None

        # Move to config or params or settings file in the long term
        self.roi_size = 32
        self.subsample = 4 # 4 is the default 
        self.long_range_scale = 32
        self.long_range_px_h = 32
        self.long_range_px_v = 32
        self.hist_bins = 16
        self.hist_start = 0
        self.dct_low_crop = 8

        # INFUTURE - Might need to come from config if these things will change
        # But for now let's just get a version working
        self.custom_model_path = './default_custom_model_sk.joblib' 
        
        self.class_to_colour = self.config.class_to_colour

    #region internal RockLens
    def high_pass(self, img, sigma):
        # From:https://stackoverflow.com/questions/50508452/implementing-photoshop-high-pass-filter-hpf-in-opencv
        # Add author zoltron
        # add date 13th Oct 2022

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        img = img - cv2.GaussianBlur(img, (0, 0), sigma) + 127
        return img

    def make_features(self,
                      img_hip,
                      img_dip,
                      img_rgb,
                      hist_bins,
                      hist_start,
                      dct_low_crop,
                      img_hsv_low_pad,
                      img_dip_low_pad,
                      long_range_scale,
                      long_range_px_h,
                      long_range_px_v,
                      start_x,
                      start_y):
        
        size_x = img_hip.shape[1] - self.roi_size
        size_y = img_hip.shape[0] - self.roi_size
        step = self.roi_size // self.subsample
        dip_channels = ('b')
        rgb_channels = ('h', 's', 'v')
        x_input = []
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        start_ = time.time()
        ii = 0
        for yy in range(0, size_y, step):
            for xx in range(0, size_x, step):

                # combined is a numpy array to hold every thing
                combined = np.zeros(
                    (dct_low_crop * dct_low_crop) +
                    (hist_bins * (len(dip_channels) + len(rgb_channels))) +
                    (128),
                    dtype=float,
                )  # TODO Need to calculate this size automatically

                # DCT of region
                img_dct = cv2.dct(img_hip[yy:yy + self.roi_size, xx:xx + self.roi_size]) / 4
                img_dct[0, 0] = 0  # Remove DC freq
                img_dct = img_dct[:dct_low_crop, :dct_low_crop]  # Get low freqs

                # Show order DCT by vertical freq first
                combined[0:dct_low_crop * dct_low_crop] = img_dct.reshape(-1, order='F')

                # Histogram of Dip for region (option for other features)
                # for i, chan in enumerate(dip_channels): - JW Changed to be a 2D, single channel dip image
                i = 0
                hist = cv2.calcHist([img_dip[yy:yy + self.roi_size, xx:xx + self.roi_size]],
                                    [i],
                                    None,
                                    [hist_bins],
                                    [hist_start, 256])
                start = (dct_low_crop * dct_low_crop) + (i * hist_bins)
                end = start + hist_bins
                combined[start:end] = (hist[:, 0]) / 1024

                # Histogram of the HSV value for the region
                for i, chan in enumerate(rgb_channels):
                    hist = cv2.calcHist([img_hsv[yy:yy + self.roi_size, xx:xx + self.roi_size]],
                                        [i],
                                        None,
                                        [hist_bins],
                                        [hist_start, 256])
                    start = (dct_low_crop * dct_low_crop) + len(dip_channels) + i * hist_bins
                    end = start + hist_bins
                    combined[start:end] = (hist[:, 0]) / 1024

                # Long range Dip, only vertical for long_range_px_v
                x_ = (start_x + xx) // long_range_scale
                y_s = (start_y + yy) // long_range_scale
                y_e = ((start_y + yy) // long_range_scale) + long_range_px_v
                # blue channel only, might need to be improved... JW - now grayscale updated.
                combined[128:160] = (img_dip_low_pad[y_s:y_e, x_].astype(float)) / 256

                # Long range RGB, only horizontal for long_range_px_h
                x_s = (start_x + xx) // long_range_scale
                x_e = ((start_x + xx) // long_range_scale) + long_range_px_h
                y_ = (start_y + yy) // long_range_scale
                combined[160:192] = (img_hsv_low_pad[y_, x_s:x_e, 0].astype(float)) / 256
                combined[192:224] = (img_hsv_low_pad[y_, x_s:x_e, 1].astype(float)) / 256
                combined[224:256] = (img_hsv_low_pad[y_, x_s:x_e, 2].astype(float)) / 256

                x_input.append(combined)

            print(f'\rCreating Features: {(yy / size_y) * 100:.0f}%', end='')
        print(f' Duration(s): {time.time() - start_}')

        # Make output a simple numpy array
        result = np.array(x_input, dtype=float)

        if result.max() > 1:
            print(f'### Warning Feature Vector Max Greater Than One: {result.max()} ###')
        if result.min() < -1:
            print(f'### Warning Feature Vector Max less Than Negative One: {result.min()} ###')

        return result

    def make_y(self, img_lab, x0, y0):
        size_x = img_lab.shape[1] - self.roi_size
        size_y = img_lab.shape[0] - self.roi_size
        step = self.roi_size // self.subsample
        y_label = []
        z_coords = []

        # INFUTUE Investigate impact of taking the average of the square at label not the centre
        offset = self.roi_size // 2  # Used to take the value at ~middle pixel of roi
        for yy in range(0, size_y, step):
            for xx in range(0, size_x, step):
                # This is where the colours need to go to a class or -1 for no class...
                # The -1 class is removed before training starts.
                if np.all(img_lab[yy + offset, xx + offset, :] == np.array(self.class_to_colour[1])):
                    y_label.append(1)  # Keep this roi
                elif np.all(img_lab[yy + offset, xx + offset, :] == np.array(self.class_to_colour[2])):
                    y_label.append(2)  # Keep this roi
                elif np.all(img_lab[yy + offset, xx + offset, :] == np.array(self.class_to_colour[3])):
                    y_label.append(3)  # Keep this roi
                elif np.all(img_lab[yy + offset, xx + offset, :] == np.array(self.class_to_colour[4])):
                    y_label.append(4)  # Keep this roi
                elif np.all(img_lab[yy + offset, xx + offset, :] == np.array(self.class_to_colour[5])):
                    y_label.append(5)  # Keep this roi
                elif np.all(img_lab[yy + offset, xx + offset, :] == np.array(self.class_to_colour[6])):
                    y_label.append(6)  # Keep this roi
                elif np.all(img_lab[yy + offset, xx + offset, :] == np.array(self.class_to_colour[7])):
                    y_label.append(7)  # Keep this roi
                elif np.all(img_lab[yy + offset, xx + offset, :] == np.array(self.class_to_colour[8])):
                    y_label.append(8)  # Keep this roi
                elif np.all(img_lab[yy + offset, xx + offset, :] == np.array(self.class_to_colour[9])):
                    y_label.append(9)  # Keep this roi
                elif np.all(img_lab[yy + offset, xx + offset, :] == np.array(self.class_to_colour[0])):
                    y_label.append(-1)  # don't care or know about the roi
                    # JW 22 Feb 2024 change to have a "don't care" class
                else:
                    y_label.append(-1)  # don't care or know about the roi

                # Calculate ii 
                xx_tile = (xx+x0) // (self.roi_size // self.subsample)
                yy_tile = (yy+y0) // (self.roi_size // self.subsample)

                # INFUTUE figure out why this goes too far
                if yy_tile > self.tile_grid.shape[0] -1:
                    yy_tile = self.tile_grid.shape[0] -1
                if xx_tile > self.tile_grid.shape[1] -1:
                    xx_tile = self.tile_grid.shape[1] -1

                ii = self.tile_grid[yy_tile, xx_tile]
                
                # keep a list of coordinates visited, xx and yy are in local crop coordinates
                # ii is converted to the tile index
                z_coords.append([xx, yy, ii])  

            print(f'\rCreating y labels: {(yy / size_y) * 100:.0f}%', end='')
        print('')

        # Make output a simple numpy array
        result = np.array(y_label, dtype=int)
        coords = np.array(z_coords, dtype=int)

        # Tested Good at this stage
        print(f'yshape {result.shape}, ymax {result.max()}, ymin {result.min()} ')
        print(f'Any Y NAN: {np.any(np.isnan(result))}')
        # input('Press enter to continue:')

        return result, coords

    def img_colour_to_class(self, img_lab):
        img_classes = np.zeros((img_lab.shape[0], img_lab.shape[1]), dtype=np.int32)

        # make this loop through all the classes in class_to_colour
        for i in self.class_to_colour:
            b,g,r = self.class_to_colour[i]
            img_classes += (np.where(img_lab[:,:,0] == b, i, 0)&\
                        np.where(img_lab[:,:,1] == g, i, 0)&\
                        np.where(img_lab[:,:,2] == r, i, 0))

        
        return img_classes

    def make_coords(self, img_shape, x0, y0):
        size_x = img_shape[1] - self.roi_size
        size_y = img_shape[0] - self.roi_size
        step = self.roi_size // self.subsample
        z_coords = []

        for yy in range(0, size_y, step):
            for xx in range(0, size_x, step):
                # Calculate ii 
                xx_tile = (xx+x0) // (self.roi_size // self.subsample)
                yy_tile = (yy+y0) // (self.roi_size // self.subsample)

                # INFUTUE figure out why this goes too far
                if yy_tile > self.tile_grid.shape[0] -1:
                    yy_tile = self.tile_grid.shape[0] -1
                if xx_tile > self.tile_grid.shape[1] -1:
                    xx_tile = self.tile_grid.shape[1] -1

                ii = self.tile_grid[yy_tile, xx_tile]
                
                # keep a list of coordinates visited, xx and yy are in local crop coordinates
                # ii is converted to the tile index
                z_coords.append([xx, yy, ii])  
            print(f'\rCreating y labels: {(yy / size_y) * 100:.0f}%', end='')
        print('')

        # Make output a simple numpy array
        coords = np.array(z_coords, dtype=int)

        return coords
   
    def predict_model(self, img_shape, z_coords):
        output = np.zeros((img_shape[0], img_shape[1]), dtype=np.int32)

        # INFUTURE investigate random prediction with decreasing size... as tested before
        low_offset = (self.roi_size // 2) - ((self.roi_size // self.subsample) // 2)
        high_offset = (self.roi_size // 2) + ((self.roi_size // self.subsample) // 2)

        start = time.time()

        # Display % complete, even if it makes things slightly slower...
        total = len(z_coords)
        for i, coord in enumerate(z_coords):
            # z_coords are [xx,yy,zz] where zz is the x_input index.
            result = self.model.predict([self.x_inputs[coord[2]]])
            output[coord[1]+low_offset:coord[1]+high_offset, coord[0]+low_offset:coord[0]+high_offset] = result[0]

            if (i % 100) == 0:
                print(f'\rCreating Model Predictions: {(i / total) * 100:.0f}%', end='')
        print('')
        print(f' Duration(sec): {time.time() - start}')

        return output

    def make_label_image(self, image_shape, labels):
        size_x = image_shape[1] - self.roi_size
        size_y = image_shape[0] - self.roi_size
        step = self.roi_size // self.subsample
        low_offset = (self.roi_size // 2) - ((self.roi_size // self.subsample) // 2)
        high_offset = (self.roi_size // 2) + ((self.roi_size // self.subsample) // 2)
        output = np.zeros((image_shape), dtype=np.uint8)
        offset = (self.roi_size // 2)

        for yy in range(0, size_y, step):
            for xx in range(0, size_x, step):
                for clr in self.class_to_colour:
                    if labels[yy + offset, xx + offset] == clr:
                        output[yy+low_offset:yy+high_offset, xx+low_offset:xx+high_offset, :] = np.array(self.class_to_colour[clr])
            print(f'\rCreating Label Image: {(yy / size_y) * 100:.0f}%', end='')
        print('')

        return output
    
    def load_model(self, path):
        self.model = joblib.load(path)
        print('load_model from path:', path)
    
    #endregion

    #region Simple methods for external calls
    def preprocess(self, img_rgb, img_dip, new_model=False):
        print(f'preprocessing | {img_rgb.shape, img_dip.shape}')

        # INFUTURE: The front end needs to know to call this again when a new image set is loaded

        # Prepare the model, new_model always forces a fresh start, mainly for qc testing
        if new_model:
            self.model = MLPClassifier(verbose=True)
        elif not self.model:
            try:
                self.model = joblib.load(self.custom_model_path)
                print('Loaded baseline model from file:', self.custom_model_path)
            except FileNotFoundError:
                self.model = MLPClassifier(verbose=True)
        

        
        start_x = 0 #int(self.images.view_width)  # 0 # original
        start_y = 0 #int(self.images.view_height) # 0 # original

        # Setup low resolution, padded, long range images
        self.img_hsv_low = cv2.resize(img_rgb,
                                None,
                                fx=1 / self.long_range_scale,
                                fy=1 / self.long_range_scale,
                                interpolation=cv2.INTER_AREA)
        self.img_hsv_low = cv2.cvtColor(self.img_hsv_low, cv2.COLOR_RGB2HSV)

        self.img_dip_low = cv2.resize(img_dip,
                                None,
                                fx=1 / self.long_range_scale,
                                fy=1 / self.long_range_scale,
                                interpolation=cv2.INTER_AREA)
        self.img_dip_low = cv2.cvtColor(self.img_dip_low, cv2.COLOR_BGR2GRAY)
        
        self.img_hsv_low_pad = np.zeros((self.img_hsv_low.shape[0],
                                    self.img_hsv_low.shape[1] + self.long_range_px_h,
                                    self.img_hsv_low.shape[2]), dtype=np.uint8)  # Pad in the horizontal direction
        
        self.img_dip_low_pad = np.zeros((self.img_dip_low.shape[0] + self.long_range_px_v,
                                    self.img_dip_low.shape[1]), dtype=np.uint8)  # Pad in the vertical direction

        self.img_hsv_low_pad[
            :self.img_hsv_low.shape[0],
            self.long_range_px_h // 2:self.img_hsv_low.shape[1] + self.long_range_px_h // 2,
            :
        ] = self.img_hsv_low

        self.img_dip_low_pad[
            self.long_range_px_v // 2:self.img_dip_low.shape[0] + self.long_range_px_v // 2,
            :self.img_dip_low.shape[1],
        ] = self.img_dip_low

        # High pass (and grayscale) image and format for DCT
        self.img_test_hip = np.float32(self.high_pass(img_rgb, 3)) / 255.0

        # Make input features (start with only the flattened low freq DCT) 
        self.x_inputs = self.make_features(
            img_hip=self.img_test_hip,
            img_dip=img_dip,
            img_rgb=img_rgb,
            hist_bins=self.hist_bins,
            hist_start=self.hist_start,
            dct_low_crop=self.dct_low_crop,
            img_hsv_low_pad=self.img_hsv_low_pad,
            img_dip_low_pad=self.img_dip_low_pad,
            long_range_scale=self.long_range_scale,
            long_range_px_h=self.long_range_px_h,
            long_range_px_v=self.long_range_px_v,
            start_x=start_x,
            start_y=start_y,
        )

        # Setup the tile grid and tile index
        x_tile_count = (img_rgb.shape[1]-self.roi_size) // (self.roi_size // self.subsample)
        y_tile_count = (img_rgb.shape[0]-self.roi_size) // (self.roi_size // self.subsample)
        self.tile_index = np.arange(0, (x_tile_count * y_tile_count), dtype=np.int64)
        self.tile_grid = self.tile_index.reshape((y_tile_count, x_tile_count))

        # If all went well
        self.preprocessing_complete = True

    def learn_predict(self, img_labels, x0, y0, x1, y1, smooth=False):
        # INFUTURE - Need mechanism to let the user know this will take some time or give interactive feeback
        
        if not self.preprocessing_complete:
            return "Preprocessing not complete. Call preproccessing before learn_predict"

        print(f'learn_predict | coords {x0, y0, x1, y1}')

        # Make the labels, new_coords is in normal orders IE [x,y] pairs
        new_labels, new_coords = self.make_y(
            img_labels, 
            x0=x0,
            y0=y0
        )

        # Add to the existing dataset - always have the newer labels at the start
        # When np.unique is run, it keeps the first occurance of a value
        self.y_labels = np.concatenate([new_labels, self.y_labels])
        self.z_coords = np.concatenate([new_coords, self.z_coords])

        # Check the labels
        # They won't match as we now do all the x_inputs at the start
        print(f'x_inputs shape {self.x_inputs.shape} {self.x_inputs.dtype}')
        print(f'y_labels shape {self.y_labels.shape} {self.y_labels.dtype}')
        print(f'z_coords shape {self.z_coords.shape} {self.z_coords.dtype}')

        # Find the labels and store them in effect removing don't care labels
        good_index_y_labels = np.where(self.y_labels != -1) # This by the index of the concat. y_labels
        all_tile_index = self.z_coords[:,2] # return the third column only
        good_index_x_inputs = all_tile_index[good_index_y_labels]
        
        y_labels_qc = self.y_labels[good_index_y_labels]
        z_coords_qc = self.z_coords[good_index_y_labels]
        x_inputs_qc = self.x_inputs[good_index_x_inputs]

        # Check the labels
        print(f'X_inputs after removing -1, shape {x_inputs_qc.shape} {x_inputs_qc.dtype}')
        print(f'y_labels after after removing -1, shape {y_labels_qc.shape} {y_labels_qc.dtype}')
        print(f'Do they match?: {"yes" if x_inputs_qc.shape[0] == y_labels_qc.shape[0] else "no"}')

        # QC to remove any duplicates (overlapping regions of input only)
        # Use the tile_index (z_coords[:,2]) to check for uniqueness
        _, good_index_z_coords = np.unique(z_coords_qc[:,2], axis=0, return_index=True)
        y_labels_no_dup = y_labels_qc[good_index_z_coords]

        # Are the x_inputs in the same order as the y labels at this point?
        x_inputs_no_dup = x_inputs_qc[good_index_z_coords]
        z_coords_no_dup = z_coords_qc[good_index_z_coords]
        print(f'X_inputs after/before removing duplicates, shape {x_inputs_no_dup.shape} {x_inputs_qc.dtype}')
        print(f'y_labels after/before removing duplicates, shape {y_labels_no_dup.shape} {y_labels_qc.dtype}')
        print(f'z_coords after/before removing duplicates, shape {z_coords_no_dup.shape} {z_coords_qc.dtype}')
        print(f'Does X_input old vs new match?: {"yes" if x_inputs_qc.shape[0] == x_inputs_no_dup.shape[0] else "no"}')
        #input('Press enter to continue:')

        # Check to see if there is NAN or max has gone crazy
        print(f'xshape {x_inputs_no_dup.shape}, xmax {x_inputs_no_dup.max()}, xmin {x_inputs_no_dup.min()} ')
        print(f'yshape {y_labels_no_dup.shape}, ymax {y_labels_no_dup.max()}, ymin {y_labels_no_dup.min()} ')
        print(f'Any X NAN: {np.any(np.isnan(x_inputs_no_dup))}')
        print(f'Any Y NAN: {np.any(np.isnan(y_labels_no_dup))}')
        # input('Press enter to continue:')

        # Learn on all the x_y inputs, not just the new ones
        print('\nStart Learning... ', end='')
        start = time.time()
        try:
            self.model.partial_fit(x_inputs_no_dup, y_labels_no_dup)
            print('Used partial fit.')
        except:
            self.model.fit(x_inputs_no_dup, y_labels_no_dup)
            print('Could not partial fit, used fit instead.')
        print(f' Duration(sec): {time.time() - start}')

        # INFUTURE think about changing where the 'new' model is saved to; writing over the default is wrong 
        joblib.dump(self.model, self.custom_model_path)
        print(f'Score: {self.model.score(x_inputs_no_dup, y_labels_no_dup)}')

        # Generate the results as a single image, using the coords from current image
        predictions = self.predict_model(
            img_labels.shape,                             
            z_coords=new_coords,
        )

        # Convert the results to the img_prediction format for display
        img_pred = self.make_label_image(
            img_labels.shape, # This is just used to get the image shape
            predictions, 
        )
    
        # blur the output
        if smooth:
            blur_size = ((self.roi_size*2) // self.subsample) - 1 #61 # Must be odd...
            img_pred = cv2.medianBlur(img_pred, blur_size)

        return img_pred

    def predict(self, x0, y0, x1, y1, smooth=False):
        # INFUTURE - Need mechanism to let the user know this will take some time or give interactive feeback
        
        # Create the image shape from coords, make 3 channel for colour output
        img_shape = (y1-y0, x1-x0, 3)

        if not self.preprocessing_complete:
            return "Preprocessing not complete. Call preproccessing before learn_predict"

        print(f'predict | coords {x0, y0, x1, y1}')

        # Make the new_coords only xx,yy in local pixels, zz is tile_index
        new_coords = self.make_coords(
            img_shape, 
            x0=x0,
            y0=y0
        )

        # Generate the results as a single image, using the coords from current image
        self.predictions = self.predict_model(
            img_shape,                             
            z_coords=new_coords
        )

        # Convert the results to the img_prediction format for display
        img_pred = self.make_label_image(
            img_shape, # This is just used to get the image shape
            self.predictions, 
        )
    
        # blur the output
        if smooth:
            blur_size = ((self.roi_size*2) // self.subsample) - 1 # Must be odd...
            img_pred = cv2.medianBlur(img_pred, blur_size)

        return img_pred
    #endregion

