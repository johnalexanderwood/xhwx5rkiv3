#from abc import ABC, abstractclassmethod
import copy
import cv2
#import rb_colour_to_name
#from enum import Enum
from functools import partial
import numpy as np
import PIL
import random
from tkinter import EventType
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.tooltip import ToolTip
#import time

from rb_types import *
#from rb_filters import Filters
from rb_plugin_base import Base
#from rb_color_picker import ColorPicker


# All plugin modules need to state order to display plugin Tool buttons on Tool menu
plugin_order = [
    'PanZoom',
    #'DrawMask', # Remove DrawMask from RockLens version until naming issue is resolved
]


class PanZoom(Base):
    # Configuration
    type_view = True       # Only changes the view 
    type_solo = True

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
        self.default_params = {
            'Active': False,         # True, False - Needed for compatability
        }

        # The values that are changes as the user interacts with the plugin
        self.params = copy.copy(self.default_params)    
        self.value = None

        self.id = id
        self.text_name = text_name
        self.is_global = True

        self.event_checkbox_change = event_checkbox_change
        super().__init__(self.params)

    def make_button_tool(self, layout, tool_parent):
        # button_select_value
        name = 'button_pan_zoom'
        image = 'arrows_gray'
        text = "Pan and Zoom"
        value = "PanZoom"
        # print(layout.ui_images)
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
        print('Start to make the widgets for the properties')

    def apply(self, mask):
        pass

    def delete_mask_or_interp(self):
        pass

    def mouse_select_value(self, event):
        self.value = self.images.screen_coords_to_hex(event.x, event.y)  
        self.widgets['label_select_value_scale'].config(text=self.value)
        print('mouse selected value', self.value) 

    # def self_mouse_motion(self, event, images, widgets):
    #     pass

    @staticmethod
    def mouse_motion(event, images, widgets):
        pass
    
    @staticmethod
    def get_cursor():
        # override cursor
        return 'fleur'
    
    def generate_id(self, config=None):
        return 'PanZoom'

    def generate_name_text(self, path=None):
        name = 'PanZoom'
        return name


class DrawMask(Base):
    # Plugin Configuration
    type_needs_apply = False
    type_draw = True
    type_mask = True 
    type_solo = True
    type_local = True
    type_create_on_tool_button = True # Plugin instance is created when the tool is created
    type_has_popup = True

    def __init__(self, 
                 config=None,
                 id=None, 
                 text_name=None,
                 is_global=False,
                 widgets=None, 
                 widget_parent=None, 
                 images=None,
                 event_scale_change=None,
                 event_checkbox_change=None):
        
        # Setup the values to create the plugin with
        # TODO push defaults to config file
        self.default_params = {
            'Active': True,         # True, False
            'Remove': True,         # True Remove, False Keep
            'RGB': True,            # True over RGB images
            'Size': 48,             # 0:1 mapped to 0 - 255
            'RGBFile': "",      # The rgb image this drawn mask applies to
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
        self.Type = DrawMask  

        # Keep  track of mouse state etc
        self.mouse = {
            'event': 0,
            'screen_x': 0, # screen_x - mouse does not know about zoom etc
            'screen_y': 0, # screen_y - mouse does not know about zoom etc
            'active': False,  # True is moving, dragging, any button down...
            'over_secondary': False,  # True if over the secondary image
            'flags': 0,
            'param': ''
        }

        super().__init__(self.params)

    def make_button_tool(self, layout, tool_parent):
        # button_select_value
        name = 'button_manual_draw'
        image = 'pencil_gray'
        text = "Draw Mask"
        value = "DrawMask"

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

        # Checkbutton keep
        self.widgets['checkbutton_manual_draw_keep'] = ttk.Checkbutton(
            self.widget_parent,
            text='Keep / Remove',
            bootstyle=(DEFAULT, TOGGLE, ROUND),
            command=partial(self.apply, True),
        )

        # TODO this should not be state but value? or something? fix
        # chk.state(['selected'])  # check the checkbox
        # chk.state(['!selected']) # clear the checkbox
        self.widgets['checkbutton_manual_draw_keep'].pack(
            side=TOP,
            ipadx=4,
            ipady=4,
            padx=1,
            pady=1,
        )

        if self.params['Remove']:
            self.widgets['checkbutton_manual_draw_keep'].state(['selected'])
        else:
            self.widgets['checkbutton_manual_draw_keep'].state(['!selected'])

        ToolTip(self.widgets['checkbutton_manual_draw_keep'], text="Keep if off. Remove if on.")       

        # Horizontal Separator
        name = 'separator_manual_draw_start'
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

        # label Size
        self.widgets['label_manual_draw_size'] = ttk.Label(
            self.widget_parent,
            text='Size',
            state=ACTIVE,
            bootstyle=(DEFAULT,),
        )

        self.widgets['label_manual_draw_size'].pack(
            self.config.standard_label
        )

        # Scale Size
        self.widgets['scale_manual_draw_size'] = ttk.Scale(
            self.widget_parent,
            value=self.params['Size'],  
            state=ACTIVE,
            bootstyle=(DEFAULT,),
            from_= 0,
            to= 255,
            command=partial(self.apply, True),
        )

        self.widgets['scale_manual_draw_size'].pack(
            self.config.standard_scale
        )

        ToolTip(self.widgets['scale_manual_draw_size'], text="Change Drawing Size")

        # if we are making the widgets we are likely going to be drawing
        self.images.draw_circle_size = int(self.params['Size']) # drawing preview circle size

    def event_popup(self, event):
        print(f'event_popup | {event}')
        # Then in plugin create plugin and populate with all the rocktype / colors...
        self.widgets['popup_menu_view'].delete(0, 'end')
        self.widgets['popup_menu_view'].add_command(
            label='Keep', 
            command=partial(self.event_popup_select, remove=False)
        ) 
        self.widgets['popup_menu_view'].add_command(
            label='Remove', 
            command=partial(self.event_popup_select, remove=True)
        ) 

        try: 
            self.widgets['popup_menu_view'].tk_popup(event.x_root, event.y_root) 
        finally: 
            self.widgets['popup_menu_view'].grab_release() 

    def event_popup_select(self, remove):
        print(f'event_popup_select | remove {remove}')

        # Set the params
        self.params['Remove'] = remove

        # ALSO force update of the widgets
        self.widgets['checkbutton_manual_draw_keep'].set(remove)
        self.widgets['checkbutton_manual_draw_keep'].update()
  
    def apply(self, changed_params=True, over_secondary=False):
        if changed_params:
            # Get the values from the scales (change to right type and range done in convert fuction)
            remove = self.widgets['checkbutton_manual_draw_keep'].instate(['selected'])
            size = int(self.widgets['scale_manual_draw_size'].get())

            self.params.update({
                'Remove': remove,
                'Size': size,
                'RGB': not over_secondary,
            })
        else:
            # Note: RGB vs Dip can only be set on the initial mouse select
            #       IE not possible to update this after creation of plugin instance.
            self.params['RGB'] = not over_secondary

        # need to send back mask
        self.images.draw_circle_size = int(self.params['Size']) # drawing preview circle size
        self.mouse_draw_line(0, 0, 0, 0, size=0, remove=True)

    def delete_mask_or_interp(self):
        # TODO
        # make this work for multiple manual drawn masks
        # if a draw mask exists delete it
        result = self.images._inter_msk.get(self.id, False)
        if hasattr(result, 'shape'): # "Tes"t" if a numpy array
            del(self.images._inter_msk[self.id])

    def mouse_select_value(self, event, over_secondary):
        print('DrawMask:mouse_select_value', event) 

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
        
        # Guard Clause: Check there are images loaded by looking at image_height
        if self.images.image_height is None:
            return
        
        # If there is no manual mask layer create it
        # TODO user the plugins id here.
        # Does the plugin know it own id at this point?
        if not self.id in self.images._inter_msk:
            self.images._inter_msk[self.id] = np.zeros((self.images.image_height, 
                                                            self.images.image_width), 
                                                            dtype=np.uint8)
            self.images._inter_msk[self.id][:,:] = self.config.blank_mask_colour

        if remove:
            colour = 255
        else:
            colour = 0

        # Correct the cursor size for the current matrix
        size = int(size / self.images.matrix[0, 0])
        if size <= 0:
            size = 1

        d_x, d_y = self.images.transform_view2buffer(old_mouse_x, old_mouse_y)
        new_d_x, new_d_y = self.images.transform_view2buffer(m_x, m_y)

        # Line needs to be more than one pixel long
        if new_d_x == d_x and new_d_y == d_y:
            new_d_x += random.choice([-1, 1])
            new_d_y += random.choice([-1, 1])

        # Draw the line on current zoom level
        try:
            cv2.line(self.images._inter_msk[self.id],
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
        print('generate_id | Not used any more.')
        return
    
    def generate_name_text(self, path):
        text_name = f"{self.Type.__name__} | {path[self.config.rgb].split('/')[-1]}"
        return text_name
    
    def prepare_save(self):
        # Overidden from base class to allow the saving of the actual numpy array
        settings = {
            'type': type(self).__name__,
            'id': self.id,
            'text_name': self.text_name,
            'params': self.params,
            'drwmsk': self.images._inter_msk[self.id]
        }

        return settings
    
    def prepare_load(self, settings):
        # Overidden from base class to allow the saving of the actual numpy array
        self.id = settings['id']
        self.text_name = settings['text_name']
        self.params = settings['params']
        self.images._inter_msk[self.id] = settings['drwmsk']



 