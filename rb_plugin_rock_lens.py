'''Plugin Class for RockLens funcationality'''
import cv2
from functools import partial
import joblib
import numpy as np
import random
from tkinter import EventType
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.tooltip import ToolTip
from ttkbootstrap.dialogs import Messagebox

import copy
import rb_plugin_base
import PIL
from rb_plugin_rock_lens_backend import RockLensBackend as Backend

# All plugin modules need to state order to display plugin Tool buttons on Tool menu
plugin_order = [
    'DrawInterp',
    'RockLens',
]

class RockLens(rb_plugin_base.Base):
    # Type Configuration
    #type_solo = True
    type_draw = False  # Allows mouse drawing (manual corrections)
    type_interp = True  # Generate a interp layer
    type_global = True  # Global plugin, affects all images
    #type_local = True
    type_create_on_tool_button = True # Plugin instance is created when the tool is created
    type_needs_apply = True

    def __init__(self, 
                 config=None,
                 id=None, 
                 text_name=None,
                 is_global=True,
                 widgets=None, 
                 widget_parent=None, 
                 images=None,
                 event_scale_change=None,
                 event_checkbox_change=None):
        
        # Setup the values to create the plugin with
        # TODO push defaults to config file
        self.default_params = {
            'Active': True,         # True, False
            'Source': None,         # Which layer are the labels / training data coming from
            'RGB': True,            # True over RGB images - Need to keep for compatability
            'WholeImage': False,    # false use current view as Region of Interest, true use whole image
            'PredictOnly': False,   # False only predict when apply is pressed, else train on labels then predict
        }

        self.config = config
        self.id = id
        self.text_name = text_name
        self.is_global = is_global
        self.widgets = widgets
        self.widget_parent = widget_parent
        self.images = images
        
        # The values that are changes as the user interacts with the plugin
        self.params = copy.copy(self.default_params)    
        self.Type = RockLens  

        # Needed for model and learning
        self.backend = Backend(
            config=config, 
        )

        super().__init__(self.params)

    def update_plugin_text_name_to_id(self):
        # Get the actual names from the plugins so they are 'more' human readable
        names = []
        plugin_text_name_to_id = {} # clear
        for plugin in self.images.plugins:
            names.append(self.images.plugins[plugin].text_name)
            plugin_text_name_to_id[names[-1]] = plugin
        
        return plugin_text_name_to_id

    def make_button_tool(self, layout, tool_parent):
        # button_select_value
        name = 'button_rock_lens_draw'
        image = 'rock_lens_gray'
        text = "RockLens"
        value = "RockLens"

        layout.ui_images[image] = layout.ui_images_raw[image].resize((
            layout.config.button_tool_width, 
            layout.config.button_tool_height
        ))
        layout.ui_images[image] = PIL.ImageTk.PhotoImage(layout.ui_images[image])

        layout.widgets[name] = ttk.Button(
            tool_parent,
            text=text,
            command=partial(layout.event_button_tool, value),
            image=layout.ui_images[image],
            bootstyle=(SECONDARY),
        )

        layout.widgets[name].pack(
            layout.config.standard_button_tool
        )

        ToolTip(layout.widgets[name], text=text)

    def make_widgets(self):
        print('Start to make the widgets for the properties: Size, keep / remove')

        # Note: The Active checkbox is made by RockBase Class

        # TODO - Experiment, forced zoom to full resolution when tool is made.
        # self.images.matrix = np.array([[1, 0, -500], [0, 1, -500]], dtype=np.float32)  # Matrix for translate / zoom

        # label Rock Type
        self.widgets['label_rock_lens_source'] = ttk.Label(
            self.widget_parent,
            text='Learning Source',
            state=ACTIVE,
            bootstyle=(DEFAULT,) 
        )

        self.widgets['label_rock_lens_source'].pack(
            self.config.standard_label
        )

        # Figure how to do the colours / rock types
        if len(self.images._inter_int) > 0:
            # Get the actual names from the plugins so they are 'more' human readable
            names = []
            self.plugin_text_name_to_id = {} # clear
            for plugin in self.images.plugins:
                names.append(self.images.plugins[plugin].text_name)
                self.plugin_text_name_to_id[names[-1]] = plugin

            # Create the combobox    
            self.widgets['combobox_rock_lens_source'] = ttk.Combobox(
                self.widget_parent,
                values=names,
                state='readonly',
            )
            self.widgets['combobox_rock_lens_source'].bind(
                '<<ComboboxSelected>>', 
                partial(self.event_combobox_change)
            )
            self.widgets['combobox_rock_lens_source'].current(0)
            self.widgets['combobox_rock_lens_source'].pack(
                self.config.standard_scale
            )
            button_learn_state = ACTIVE  # There interps. so ok to learn
        else:
            names = ['No Interps. To Learn From']
            self.widgets['combobox_rock_lens_source'] = ttk.Combobox(
                self.widget_parent,
                values=names,
                state=DISABLED,
            )
            # No command as combobox is disabled.
            self.widgets['combobox_rock_lens_source'].current(0)
            self.widgets['combobox_rock_lens_source'].pack(
                self.config.standard_scale
            )
            button_learn_state = DISABLED  # There no interps. so nothing to learn from

        # Horizontal Separator
        name = 'separator_rock_lens_mid_0'
        self.widgets[name] = ttk.Separator(
            self.widget_parent,
            orient=HORIZONTAL,
            bootstyle=(DEFAULT),
        )
        self.widgets[name].pack(
            side=TOP, 
            fill=X,
            ipady=4,
            pady=1,
        )

        # Current View (or small Region Of Interest) versus Whole Image Learning Prediction
        self.widgets['checkbutton_rock_lens_view_whole'] = ttk.Checkbutton(
            self.widget_parent,
            text='Current View / Whole Image',
            bootstyle=(DEFAULT, TOGGLE, ROUND)
        )
        self.widgets['checkbutton_rock_lens_view_whole'].pack(
            side=TOP,
            ipadx=4,
            ipady=4,
            padx=1,
            pady=1,
        )

        if self.params['WholeImage']:
            self.widgets['checkbutton_rock_lens_view_whole'].state(['selected'])
        else:
            self.widgets['checkbutton_rock_lens_view_whole'].state(['!selected'])

        ToolTip(self.widgets['checkbutton_rock_lens_view_whole'], text="Use current view only if off, or whole image if on.")

        # Horizontal Separator
        name = 'separator_rock_lens_mid_1'
        self.widgets[name] = ttk.Separator(
            self.widget_parent,
            orient=HORIZONTAL,
            bootstyle=(DEFAULT),
        )
        self.widgets[name].pack(
            side=TOP, 
            fill=X,
            ipady=4,
            pady=1,
        )

        # Learn then Predict on area Versus Predict only. (Region set above)
        self.widgets['checkbutton_rock_lens_predict_only'] = ttk.Checkbutton(
            self.widget_parent,
            text='Learn and Predict / Predict Only',
            bootstyle=(DEFAULT, TOGGLE, ROUND)
        )
        self.widgets['checkbutton_rock_lens_predict_only'].pack(
            side=TOP,
            ipadx=4,
            ipady=4,
            padx=1,
            pady=1,
        )
        if self.params['PredictOnly']:
            self.widgets['checkbutton_rock_lens_predict_only'].state(['selected'])
        else:
            self.widgets['checkbutton_rock_lens_predict_only'].state(['!selected'])

        ToolTip(self.widgets['checkbutton_rock_lens_predict_only'], text='Either learn from labels and then predict if off, or predict only if on.')

        # Horizontal Separator
        name = 'separator_rock_lens_mid_1'
        self.widgets[name] = ttk.Separator(
            self.widget_parent,
            orient=HORIZONTAL,
            bootstyle=(DEFAULT),
        )
        self.widgets[name].pack(
            side=TOP, 
            fill=X,
            ipady=4,
            pady=1,
        )

        # # Button Learn
        # self.widgets['button_rock_lens_learn'] = ttk.Button(
        #     self.widget_parent,
        #     text="Learn Interp.", 
        #     command=partial(self.event_learn),
        #     image=None, 
        #     state=button_learn_state,
        #     bootstyle=DEFAULT,
        # )
        # self.widgets['button_rock_lens_learn'].pack(
        #     self.config.standard_scale
        # )

        # ToolTip(self.widgets['button_rock_lens_learn'], text="Learn from Existing Interpertations")

        # # Button Predict View Only
        # self.widgets['button_rock_lens_predict_view'] = ttk.Button(
        #     self.widget_parent,
        #     text="Interp. Predict View", 
        #     command=partial(self.event_predict_view), 
        #     image=None, 
        #     state=button_learn_state,
        #     bootstyle=DEFAULT,
        # )
        # self.widgets['button_rock_lens_predict_view'].pack(
        #     self.config.standard_scale
        # )
        # ToolTip(
        #     self.widgets['button_rock_lens_predict_view'], 
        #     text="Create an Interpertation Prediction For Current View.",
        # )

        # # Button Predict Whole Image Only
        # self.widgets['button_rock_lens_predict_image'] = ttk.Button(
        #     self.widget_parent,
        #     text="Interp. Predict Image", 
        #     command=partial(self.event_predict_image), 
        #     image=None, 
        #     state=ACTIVE,
        #     bootstyle=DEFAULT,
        # )
        # self.widgets['button_rock_lens_predict_image'].pack(
        #     self.config.standard_scale
        # )
        # ToolTip(
        #     self.widgets['button_rock_lens_predict_view'], 
        #     text="Create an Interpertation Prediction For Whole Image.",
        # )

    def event_combobox_change(self, event):
        # Can we get the selected item
        selected = self.widgets['combobox_rock_lens_source'].get()
        print(f"event_combobox_change  | {event} | selected: {selected}")

        # Then we can add this to the params
        self.params['Source'] = selected

    def event_learn(self):
        pass
        # print(f'event_learn |')
        
        # # TODO - Might need to come from config if these things will change
        # # But for now let's just get a version working
        # custom_model_path = './default_custom_model_sk.joblib' 
        # if not self.model:
        #     self.model = joblib.load(custom_model_path)
        #     print('Loaded baseline model from file:', custom_model_path)
        
        # # Original configuration as what was shown to Norway
        # feature_selector = None
        
        # # TODO make this the current view

        # learn_from_view = True # True # alternative is learn_from_whole
        # if learn_from_view:
                
        #         img_rgb = cv2.warpAffine(
        #             self.images._in[self.config.rgb], 
        #             self.images.matrix, 
        #             (int(self.images.view_width), int(self.images.view_height)),
        #             flags=self.images.interpolations[self.config.interpolation_type]
        #         )

        #         img_dip = cv2.warpAffine(
        #             self.images._in[self.config.dip], 
        #             self.images.matrix, 
        #             (int(self.images.view_width), int(self.images.view_height)),
        #             flags=self.images.interpolations[self.config.interpolation_type]
        #         )
        # else:
        #     img_rgb = self.images._in[self.config.rgb]
        #     img_dip = self.images._in[self.config.dip]
        
        
        # roi_size = 32
        # subsample = 4 
        # long_range_scale = 32
        # long_range_px_h = 32
        # long_range_px_v = 32
        # start_x = int(self.images.view_width)  # 0 # original
        # start_y = int(self.images.view_height) # 0 # original
            
        # # This does not work just now: With smaller input images and 8x8 roi
        # # feature_selector = None
        # # img_rgb = cv2.resize(self.images._in[self.config.rgb], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        # # img_dip = cv2.resize(self.images._in[self.config.dip], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        # # roi_size = 8
        # # subsample = 4 
        # # long_range_scale = 8
        # # long_range_px_h = 8
        # # long_range_px_v = 8
        # # start_x = 0
        # # start_y = 0

        # # Setup low resolution, padded, long range images
        # img_rgb_low = cv2.resize(img_rgb,
        #                         None,
        #                         fx=1 / long_range_scale,
        #                         fy=1 / long_range_scale,
        #                         interpolation=cv2.INTER_AREA)
        # img_rgb_low = cv2.cvtColor(img_rgb_low, cv2.COLOR_RGB2HSV)
        # img_dip_low = cv2.resize(img_dip,
        #                         None,
        #                         fx=1 / long_range_scale,
        #                         fy=1 / long_range_scale,
        #                         interpolation=cv2.INTER_AREA)
        
        # img_rgb_low_pad = np.zeros((img_rgb_low.shape[0],
        #                             img_rgb_low.shape[1] + long_range_px_h,
        #                             img_rgb_low.shape[2]), dtype=np.uint8)  # Pad in the horizontal direction
        # img_dip_low_pad = np.zeros((img_dip_low.shape[0] + long_range_px_v,
        #                             img_dip_low.shape[1]), dtype=np.uint8)  # Pad in the vertical direction

        # class_to_colour = self.config.class_to_colour

        # if self.params['Source'] == None:
        #     # Return if there is no draw interp layer avaliable
        #     Messagebox.show_error('No Interp. Layer Selected', 'No Interp. Layer Selected \n' + 'Please select layer it before learning.')
        #     return 

        # text_name = self.params['Source']
        # plugin_id = self.plugin_text_name_to_id[text_name]
        # print(f'Learning Source is text_name:"{self.params["Source"]}", id: {plugin_id}')

        # if plugin_id in self.images._inter_int:
        #     img_labels = self.images._inter_int[plugin_id]
        # else:
        #     # Return if there is no draw interp layer avaliable
        #     Messagebox.show_error('Layer Error!', 'Layer Error! \n' + 'Draw Interp. does not exist. Please create it before learning.')
        #     return 
        
        # # TODO
        # # Need mechanism to let the user know this will take some time
        # # Really update the busy meter on the main form, but that is not know to this 
        # # plugin.
        # # Can the methods be passed to this plugin?
        # # Or can we have a pop up?

        # # image keep was rename image_labels 
        # # image keep is the image the user has annotated - in RockBase this from the DrawInterp Tool
        # # or more correctly where ever the user has selected from the combobox...
        # # it could also come from other plugins that generate a interp layer (larger / more complex models?)
        # img_pred, self.model, self.x_inputs, self.y_labels, self.z_coords = backend.learn_process(
        #     self.model,
        #     feature_selector,
        #     custom_model_path,
        #     img_rgb,
        #     img_dip,
        #     img_labels,
        #     self.x_inputs,
        #     self.y_labels,
        #     self.z_coords,
        #     roi_size,
        #     subsample,
        #     img_rgb_low_pad,
        #     img_dip_low_pad,
        #     long_range_scale,
        #     long_range_px_h,
        #     long_range_px_v,
        #     start_x,
        #     start_y,
        #     class_to_colour,
        #     smooth_masks=True,
        # )

        # # blur the output
        # # blur_size = 61 # Must be odd...
        # # img_pred = cv2.medianBlur(img_pred, blur_size)

        # # Original 32 Roi and full size
        # if learn_from_view:
        #     offset_x = self.images.matrix[0,2]
        #     offset_y = self.images.matrux[1,2]
        #     self.images._inter_int[self.id][offset_x:int(self.images.view_height), offset_y:int(self.images.view_width)] = img_pred
        # else:
        #     self.images._inter_int[self.id] = img_pred
       
        # # For smaller prediction being resized up
        # #self.images._inter_int[self.id] = cv2.resize(img_pred, None, fx=4, fy=4, interpolation=cv2.INTER_AREA)

        # # As there is a new predicted interp the view needs to be updated by timer.
        # self.images.needs_update = True

    def event_predict_view(self):
        print(f'event_predict_view | ')

        # Add prediction to the RockLens interp layer

    # JW - Change of mental model for this plugin
    # Everything happens through the apply button.
    # The checkboxes (toggles) are using to change the modes the plugin is in
    # So there are:
    #   • Use the current view (typically small/zoomed in)
    #   • Use the whole image (IE the full resolution of the image set)
    #   • Learn from a designated layer and predict - Manual drawing/labels have priority 
    #   • Predict only (that is use existing training)
    #   • Finally we need to think about what happens to the prediction
    #       • Conceptually it would be best to match how the masks work
    #       • Therefore, ultimately there is only one stacked label image shown to the user...
    #       • The labels should have a priotiy, it any thing manually drawn has precendance.
    #       • The user is always right...

    # TODO
    # Note: No - there is one combined interp/obs/pred of stacked outputs...
    # Therefore we don't need the system below - leave this as way of explaination 
    # make a slider which say output predictions to new layer or source layer...
    # Where do we write the image to the interp layer or to a new layer or to the RockClean layer?

    def apply(self, changed_params=True, over_secondary=False):
        print(f'apply | changed_params {changed_params}')

        # JW - Debug - just a test
        if not changed_params:
            print(f'apply | changed_params {changed_params} |  returning from function...')
            return
        elif changed_params:
            # Get the new params:
            #   whole_image:    True, work with whole image. False only do current view.
            #   predict_only:   True, only do prediction. False, first learn then do prediction.
            #   source:         Which plugin to get the labels / interp / obs from
            whole_image = self.widgets['checkbutton_rock_lens_view_whole'].instate(['selected'])
            predict_only = self.widgets['checkbutton_rock_lens_predict_only'].instate(['selected'])
            source = self.widgets['combobox_rock_lens_source'].get()

            # Store for later
            self.params.update({
                'WholeImage': whole_image,
                'PredictOnly': predict_only,
                'Source': source,
            })

        # if in current view mode, limit working region to current view
        if not self.params['WholeImage']: 
            # This finds the top left coords - 0, 0 is in view coordinates
            x0, y0 = self.images.transform_view2buffer(0,0)

            x0, y0 = x0 - self.backend.roi_size, y0 - self.backend.roi_size

            x0, y0 = self.images.limit_inbounds(x0, y0)
            print(f'apply | screen 0,0 in image coords {x0, y0}')
        
            # This finds the bottom right coords and keep inbounds
            x1 = self.widgets["label_view_upper"].winfo_width()
            y1 = self.widgets["label_view_upper"].winfo_height()
            x1, y1 = self.images.transform_view2buffer(x1, y1)

            x1, y1 = x1 + self.backend.roi_size, y1 + self.backend.roi_size

            x1, y1 = self.images.limit_inbounds(x1, y1)
            print(f'apply | "view" coords {x0, y0, y1, x1}')

            # Expand the image if possible

        else:
            x0, y0 = 0, 0
            y1, x1, c = self.images._in[self.config.rgb].shape
            print(f'apply | full image {x0, y0, y1, x1}')

        if not self.backend.preprocessing_complete:
            # preprocess does everything does not have to be run for new labels 
            self.backend.preprocess(
                img_rgb=self.images._in[self.config.rgb],
                img_dip=self.images._in[self.config.dip],
            )

        # Get the source of the labels as an id, learn from it and predict result
        conversion = self.update_plugin_text_name_to_id()
        source_id = conversion[self.params['Source']]

        if not self.params['PredictOnly']:
            img_pred = self.backend.learn_predict(
                self.images._inter_int[source_id][y0:y1, x0:x1,:], 
                x0, 
                y0, 
                x1, 
                y1,
                smooth=True,
            )
        else: 
            # Predict only on the view or whole image as per coordinates
            img_pred = self.backend.predict(x0, y0, x1, y1, smooth=True,)
        
        # Add the new prediction to the dictionary using the source image plugin id as name
        # Put the new prediction into the correct part and don't overwrite any existing manual stuff
        self.images._inter_int[source_id][y0:y1, x0:x1] = self.merge_prediction(
            img_org=self.images._inter_int[source_id][y0:y1, x0:x1], 
            img_pre=img_pred,    
        )

        # As there is a new predicted interp the view needs to be updated by timer.
        self.images.needs_update = True

    def merge_prediction(self, img_org, img_pre):
        mask = cv2.inRange(img_org, (0,0,0), (0,0,0))
        img_pre_masked = cv2.bitwise_and(img_pre, img_pre, mask=mask)
        return img_org + img_pre_masked


    def delete_mask_or_interp(self):
        # RockLens should draw to a "RockLens" Interp. Layer...
        result = self.images._inter_int.get('RockLens', False)
        if hasattr(result, 'shape'):
            del(self.images._inter_int['RockLens'])

    def mouse_select_value(self, event, over_secondary):
        pass

    @staticmethod
    def mouse_motion(event, images, widgets):
        pass

    @staticmethod
    def get_cursor():
        # override cursor
        return ''
    
    @staticmethod
    def generate_id(config):
        # Quit as this is a mistake if called...
        print('RockLens should not generate its own id, exiting...')
        exit()
    
    def generate_name_text(self, path=None):
        # Needs to have path if create long form name, 
        # which is in turn needed for local plugins
        # JW - Add Global version for better functionality
        if not path:
            text_name = f"{self.Type.__name__} | Global {self.id}"
        else:
            text_name = f"{self.Type.__name__} | {path[self.config.rgb].split('/')[-1]}"
        
        return text_name
    

class DrawInterp(rb_plugin_base.Base):
    # Type Configuration
    type_solo = True
    type_draw = True                    # Allows mouse drawing (manual corrections)
    type_interp = True                  # Generate a interp layer
    type_local = True                   # Global plugin, affects all images
    type_create_on_tool_button = True   # Plugin instance is created when the tool is created
    type_needs_apply = False            # Plugin only has delete button as runs live
    type_has_popup = True               # Plugin generates a popup menu - for selecting rocktypes

    def __init__(self, 
                 config,
                 id=None, 
                 text_name=None,
                 is_global=False,
                 widgets=None, 
                 widget_parent=None, 
                 images=None,
                 event_scale_change=None,
                 event_checkbox_change=None):
        
        self.config = config
        self.id = id
        self.text_name = text_name
        self.is_global = is_global
        self.widgets = widgets
        self.widget_parent = widget_parent
        self.images = images

        # The values that are changes as the user interacts with the plugin
        # Setup the values to create the plugin with
        # TODO push defaults to config file
        self.default_params = {
            'Active': True,         # True, False
            'Remove': True,         # True Remove, False Keep
            'RGB': True,            # True over RGB images
            'RockType': self.config.default_rock_type, # Which rock class or type (from dict/scheme/book is in use )
            'Size': 48,             # 0 - 255
            'RGBFile': "",          # RGB file drawn interp. applies to 
        }
        self.params = copy.copy(self.default_params)    

        self.Type = DrawInterp  # Can this be removed? Where is this value used?

        # Keep  track of mouse state etc
        self.mouse = {
            'event': 0,
            'screen_x': 0, #1389//2,  # screen_x - mouse does not know about zoom etc
            'screen_y': 0, #429//2,  # screen_y - mouse does not know about zoom etc
            'active': False,  # True is moving, dragging, any button down...
            'over_secondary': False,  # True if over the secondary image
            'flags': 0,
            'param': ''
        }

        super().__init__(self.params)

    def make_button_tool(self, layout, tool_parent):
        # button_select_value
        name = 'button_draw_interp_draw'
        image = 'pencil_interp'
        text = "Draw Interp."
        value = "DrawInterp"

        layout.ui_images[image] = layout.ui_images_raw[image].resize((
            layout.config.button_tool_width, 
            layout.config.button_tool_height
        ))
        layout.ui_images[image] = PIL.ImageTk.PhotoImage(layout.ui_images[image])

        layout.widgets[name] = ttk.Button(
            tool_parent,
            text=text,
            command=partial(layout.event_button_tool, value),
            image=layout.ui_images[image],
            bootstyle=(SECONDARY),
        )

        layout.widgets[name].pack(
            layout.config.standard_button_tool
        )

        ToolTip(layout.widgets[name], text=text)

    def make_widgets(self):
        print('Start to make the widgets for the properties: Size, keep / remove')

        # Note: The Active checkbox is made by RockBase Class

        # label Rock Type
        self.widgets['label_draw_interp_rock_type'] = ttk.Label(
            self.widget_parent,
            text='Rock Type',
            state=ACTIVE,
            bootstyle=(DEFAULT,) 
        )

        self.widgets['label_draw_interp_rock_type'].pack(
            self.config.standard_label
        )

        # Figure how to do the colours / rock types
        names = list(self.config.rock_types.keys())

        # TODO try menubutton over combobox - we don't need to write in the box
        # ttk.Menubutton(

        # )

        self.widgets['combobox_draw_interp_type'] = ttk.Combobox(
            self.widget_parent,
            values=names,
            state='readonly',
        )
        self.widgets['combobox_draw_interp_type'].bind('<<ComboboxSelected>>', partial(self.apply, True))
        self.widgets['combobox_draw_interp_type'].current(0)
        self.widgets['combobox_draw_interp_type'].pack(
            #fill=X, 
            side=TOP,
            ipadx=1,
            ipady=1,
            padx=1,
            pady=1,
        )

        # Horizontal Separator
        name = 'separator_draw_interp_start'
        self.widgets[name] = ttk.Separator(
            self.widget_parent,
            orient=HORIZONTAL,
            bootstyle=(DEFAULT),
        )
        self.widgets[name].pack(
            side=TOP, 
            fill=X,
            #ipadx=4,
            ipady=4,
            #padx=1,
            pady=1,
        )

        # label size
        self.widgets['label_draw_interp_size'] = ttk.Label(
            self.widget_parent,
            text='Size',
            state=ACTIVE,
            bootstyle=(DEFAULT,),
        )

        self.widgets['label_draw_interp_size'].pack(
            self.config.standard_label
        )

        # Scale Size
        self.widgets['scale_draw_interp_size'] = ttk.Scale(
            self.widget_parent,
            value=self.params['Size'],  
            state=ACTIVE,
            bootstyle=(DEFAULT,),
            from_=0,
            to=255,
            command=partial(self.apply, True),
        )

        self.widgets['scale_draw_interp_size'].pack(
            self.config.standard_scale
        )

        ToolTip(self.widgets['scale_draw_interp_size'], text="Change Drawing Size")

        # Horizontal Separator
        name = 'separator_draw_interp_mid_1'
        self.widgets[name] = ttk.Separator(
            self.widget_parent,
            orient=HORIZONTAL,
            bootstyle=(DEFAULT),
        )
        self.widgets[name].pack(
            side=TOP, 
            fill=X,
            ipady=4,
            pady=1,
        )

        # Button Crop To Mask
        self.widgets['button_draw_interp_crop'] = ttk.Button(
            self.widget_parent,
            text="Crop To Mask", 
            command=partial(self.event_crop_to_mask), 
            image=None, 
            state=ACTIVE,
            bootstyle=DEFAULT,
        )

        self.widgets['button_draw_interp_crop'].pack(
            self.config.standard_scale
        )

        ToolTip(self.widgets['button_draw_interp_crop'], text="Crop Interpretation To Mask")

        # if we are making the widgets we are likely going to be drawing
        self.images.draw_circle_size = int(self.params['Size']) # drawing preview circle size

    def event_crop_to_mask(self):
        print(f'event_crop_to_mask')

        if self.config.drwmsk in self.images._in:

            # Force the loaded mask on to the current interp file
            self.images._inter_int[self.id][:,:,0] = np.where(
                self.images._in[self.config.drwmsk]==0, 
                self.images._inter_int[self.id][:,:,0], 
                0
            )
            self.images._inter_int[self.id][:,:,1] = np.where(
                self.images._in[self.config.drwmsk]==0, 
                self.images._inter_int[self.id][:,:,1], 
                0
            )
            self.images._inter_int[self.id][:,:,2] = np.where(
                self.images._in[self.config.drwmsk]==0, 
                self.images._inter_int[self.id][:,:,2], 
                0
            )

            self.widgets['label_status_plugins'].text = "Interpretation cropped to mask."
            self.images.needs_update = True
        else:
            self.widgets['label_status_plugins'].text = "No masks avaliable to crop interpretation to."
        print(f'event_crop_to_mask')

    def event_popup(self, event):
        print(f'event_popup | {event}')
        # Then in plugin create plugin and populate with all the rocktype / colors...
        self.widgets['popup_menu_view'].delete(0, 'end')
        for rock_type in self.config.rock_types:
            self.widgets['popup_menu_view'].add_command(
                label=rock_type, 
                command=partial(self.event_popup_select, rock_type)
            ) 

        try: 
            self.widgets['popup_menu_view'].tk_popup(event.x_root, event.y_root) 
        finally: 
            self.widgets['popup_menu_view'].grab_release() 

    def event_popup_select(self, rock_type):
        print(f'event_popup_select | {rock_type}')

        # Set the params
        self.params['RockType'] = rock_type

        # ALSO force update of the widgets
        self.widgets['combobox_draw_interp_type'].set(rock_type)
        self.widgets['combobox_draw_interp_type'].update()
  
    def apply(self, changed_params=True, over_secondary=False):
        print(f'{self.Type} | Apply Clicked {self.Type}')
        if changed_params:
            # Get the values from the scales (change to right type and range done in convert fuction)
            rock_type = self.widgets['combobox_draw_interp_type'].get()
            size = self.widgets['scale_draw_interp_size'].get()

            self.params.update({
                'Size': size,
                'RockType': rock_type,
            })

        self.images.draw_circle_size = int(self.params['Size']) # drawing preview circle size
        self.mouse_draw_line(0, 0, 0, 0, size=0, remove=True)  # Use this to force interp update
        
        # As the crop to mask happens inside plugins, App does not know to refresh
        # Therefore force update
        self.images.needs_update = True

    def delete_mask_or_interp(self):
        result = self.images._inter_int.get(self.id, False)
        if hasattr(result, 'shape'):
            del(self.images._inter_int[self.id])

    def mouse_select_value(self, event, over_secondary):
        print('RockLens: mouse_select_value', event) 

    def mouse_drawing(self, event):
        size = self.params['Size']
        remove = self.params['Remove']

        if event.type == EventType.ButtonPress and event.num == 1:
            # Draw line with current class
            self.mouse_draw_line(
                event.x, 
                event.y,
                self.mouse['screen_x'],
                self.mouse['screen_y'],
                size,
                remove,
            )
            self.mouse['active'] = True
        elif event.type == EventType.Motion and self.mouse['active'] is True:
            # Draw line with current class
            self.mouse_draw_line(
                event.x, 
                event.y,
                self.mouse['screen_x'],
                self.mouse['screen_y'],
                size,
                remove,
            )
        elif event.type == EventType.ButtonRelease and event.num == 1:
            # reset
            self.mouse['active'] = False

        # keep the x and y
        self.mouse['screen_x'] = event.x
        self.mouse['screen_y'] = event.y

    def mouse_draw_line(self, m_x, m_y, old_mouse_x, old_mouse_y, size, remove):
        """draw on the screen - continuous draw for now"""
        print(f"{self.Type}: mouse_draw_line")
        
        # Guard Clause: Check there are images loaded by looking at image_height
        if self.images.image_height is None:
            return

        colour = self.config.rock_types[self.params['RockType']]

        # Correct the cursor size for the current matrix
        size = int(size / self.images.matrix[0, 0])
        if size <= 0:
            size = 1

        d_x, d_y = self.images.transform_view2buffer(old_mouse_x, old_mouse_y)
        new_d_x, new_d_y = self.images.transform_view2buffer(m_x, m_y)

        if new_d_x == d_x and new_d_y == d_y:
            new_d_x += random.choice([-1, 1])
            new_d_y += random.choice([-1, 1])

        # If there is no manual mask layer create it
        if not self.id in self.images._inter_int:
            self.images._inter_int[self.id] = np.zeros((self.images.image_height, 
                                                                   self.images.image_width,
                                                                   self.images.image_channels), 
                                                                   dtype=np.uint8)
            self.images._inter_int[self.id][:,:] = self.config.blank_interp_colour

        # draw the line on current zoom level
        try:
            cv2.line(self.images._inter_int[self.id],
                     (d_x, d_y),
                     (new_d_x, new_d_y),
                     colour,
                     size)
        except cv2.error:
            # This error should no longer happen, but just incase message added
            self.print(f'Drawing error')
    
    @staticmethod
    def mouse_motion(event, images, widgets):
        pass

    @staticmethod
    def get_cursor():
        # override cursor
        return 'pencil'
    
    @staticmethod
    def generate_id(config):
        return config.drwint
    
    def generate_name_text(self, path=None):
        # Needs to have path if create long form name, 
        # which is in turn needed for local plugins
        if not path:
            text_name = f"{self.Type.__name__} | Global {self.id}"
        else:
            text_name = f"{self.Type.__name__} | {path[self.config.rgb].split('/')[-1]}"
        
        return text_name
    
