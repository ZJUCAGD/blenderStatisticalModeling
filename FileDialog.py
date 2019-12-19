import bpy
import cv2
import os 
from bpy.props import StringProperty, BoolProperty 
from bpy_extras.io_utils import ImportHelper 
from bpy.types import Operator 


bpy.types.Scene.filepath = bpy.props.StringProperty(name="FilePath", default="Unknown")
class OT_TestOpenFilebrowser(Operator, ImportHelper): 
    bl_idname = "test.open_filebrowser" 
    bl_label = "Choose a texture as hollow pattern" 
    filter_glob : StringProperty(default='*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp', options={'HIDDEN'}) 
    some_boolean : BoolProperty(name='Do a thing', description='Do a thing with the file you\'ve selected', default=True, ) 
    def execute(self, context): 
        """Do something with the selected file(s).""" 
        filename, extension = os.path.splitext(self.filepath)
        context.scene.filepath = self.filepath
        img = cv2.imread(self.filepath, 0)
        cv2.imshow('img', img)
        print('Selected file:', self.filepath) 
        print('File name:', filename) 
        print('File extension:', extension) 
        print('Some Boolean:', self.some_boolean) 
        return {'FINISHED'} 
def register(): 
    bpy.utils.register_class(OT_TestOpenFilebrowser) 
def unregister(): 
    bpy.utils.unregister_class(OT_TestOpenFilebrowser) 

if __name__ == "__main__": 
    register()
    
# test call 
# bpy.ops.test.open_filebrowser('INVOKE_DEFAULT')