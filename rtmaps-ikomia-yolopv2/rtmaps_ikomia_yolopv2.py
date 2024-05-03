# Copyright (C) 2024 Intempora SAS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rtmaps.core as rt
import rtmaps.types
from rtmaps.base_component import BaseComponent  # base class

from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    
    # Constructor has to call the BaseComponent parent class
    def __init__(self):
        BaseComponent.__init__(self)  # call base class constructor

    # Dynamic is called frequently:
    # - When loading the diagram
    # - When connecting or disconnecting a wire
    # Here you create your inputs, outputs and properties
    def Dynamic(self):
        # Adding an input called "img_in" with type IPL image (suitable for BGR images for instance)
        self.add_input("img_in", rtmaps.types.IPL_IMAGE)  # define an input

        # Define the properties and outputs. 
        self.add_property("detect_objects",True)

        self.detect_objects = self.get_property("detect_objects")
        self.output_img_w_graphics = False
        if (self.detect_objects):
            self.add_output("objs_bbox", rtmaps.types.DRAWING_OBJECT, 512) # output for up to 512 bounding boxes.
            self.add_output("objs_labels", rtmaps.types.DRAWING_OBJECT, 512) # output for up to 512 labels for bounding boxes.
            self.add_property("output_image_with_graphics",False)
            self.output_img_w_graphics = self.get_property("output_image_with_graphics") # do you need an output with objects already overlaid? (you can also do that separately in RTMaps as shown in the sample diagram)?
            if (self.output_img_w_graphics):
                self.add_output("img_out", rtmaps.types.IPL_IMAGE)

        self.add_property("detect_road_lanes",True)
        self.detect_road_lanes = self.get_property("detect_road_lanes")
        if (self.detect_road_lanes):
            self.add_output("mask", rtmaps.types.IPL_IMAGE)
        
        # add properties for model parameters    
        self.add_property("conf_threshold",0.2)
        self.add_property("intersection_over_union_threshold",0.45)
        
        
# Birth() will be called once at diagram execution startup
    def Birth(self):
        self.wf = Workflow() # create the Ikomia workflow
        self.algo = self.wf.add_task(name="infer_yolop_v2", auto_connect=True) # Create the YoloP v2 model task in the workflow
        self.first_time = True


# Core() is called every time you have a new inputs available, depending on your chosen reading policy
    def Core(self):
        # Just copy the input to the output here
        ipl_image = self.inputs["img_in"].ioelt.data
        frame = ipl_image.image_data

        if (self.first_time):
            self.first_time = False
            # When first image is received, we can initialize the model parameters (including image size)
            self.algo.set_parameters({
                "input_size": str(frame.shape[0]),
                "conf_thres": str(self.get_property("conf_threshold")),
                "iou_thres": str(self.get_property("intersection_over_union_threshold")),
                "object": str(self.detect_objects),
                "road_lane": str(self.detect_road_lanes)
            })

        # Execute the model on the incoming image.
        self.wf.run_on(frame)

        # Output road lane masks if road lanes detection is activated.
        if (self.detect_objects):
            objs = rtmaps.types.Ioelt()
            labels = rtmaps.types.Ioelt()
            objs.data = []
            labels.data = []
            count = 0
            for object in self.algo.get_output(1).get_objects():
                objs.data.append(rtmaps.types.DrawingObject())  # Drawing Object
                objs.data[count].kind = 2  # Spot = 0, Line = 1, Rectangle = 2, Circle = 3, Ellipse = 4, Text = 5
                objs.data[count].id = object.id
                objs.data[count].color = object.color[2] + (object.color[1] << 8) + (object.color[0] << 16)  # Color of the object
                objs.data[count].width = 3
                objs.data[count].misc1 = 0
                objs.data[count].misc2 = 0
                objs.data[count].misc3 = 0
                objs.data[count].data = rtmaps.types.Rectangle()  # Drawing Object is a line
                objs.data[count].data.x1 = object.box[0]
                objs.data[count].data.y1 = object.box[1]
                objs.data[count].data.x2 = object.box[0] + object.box[2]
                objs.data[count].data.y2 = object.box[1] + object.box[3]

                labels.data.append(rtmaps.types.DrawingObject())
                labels.data[count].kind = 5
                labels.data[count].id = object.id
                labels.data[count].color = objs.data[count].color
                labels.data[count].width = 2
                labels.data[count].misc1 = 0
                labels.data[count].misc2 = 0
                labels.data[count].misc3 = 0
                labels.data[count].data = rtmaps.types.Text()
                labels.data[count].data.x = objs.data[count].data.x1
                labels.data[count].data.y = objs.data[count].data.y1
                labels.data[count].data.cwidth = 5
                labels.data[count].data.cheight = 5
                labels.data[count].data.orientation = 0
                labels.data[count].data.bkcolor = 0
                labels.data[count].data.text = object.label

                count += 1

            self.write("objs_bbox", objs)  # and write to the output
            self.write("objs_labels", labels) # and write to the output

            # Output the image with overlaid objects if requested.
            if (self.output_img_w_graphics):
                out = rtmaps.types.Ioelt()
                out.data = rtmaps.types.IplImage()
                out.data.image_data = self.algo.get_image_with_graphics()
                out.data.color_model = "COLR"  # Color model can be : MONO or COLR (see C++ SDK reference for more details)
                out.data.channel_seq = "BGR"  # Channel sequence can also be : RGB, BGR, RGBA, BGRA, YUV, YUVA, GRAY... (see C++ SDK reference for more details)
                self.write("img_out", out)

        # Output road lanes mask if road lanes detection was activated.
        if (self.detect_road_lanes):
            mask = rtmaps.types.Ioelt()
            mask.data = rtmaps.types.IplImage()
            mask.data.image_data = self.algo.get_output(0).get_overlay_mask()
            mask.data.color_model = "COLR"
            mask.data.channel_seq = "BGRA"
            self.write("mask",mask)



# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass
