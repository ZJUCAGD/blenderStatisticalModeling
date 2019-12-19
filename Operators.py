import bpy

class RandomSample(bpy.types.Operator):
    """Sample discrete Guassian process"""
    bl_idname = "test.randomsample"
    bl_label = "Random Sample Deformation"
    bl_options = {'REGISTER', 'UNDO'}


    def execute(self, context):
        print('Random Sample!!')
        return {'FINISHED'}
class Reset(bpy.types.Operator):
    """Reset"""
    bl_idname = "test.reset"
    bl_label = "Reset Deformation"
    bl_options = {'REGISTER', 'UNDO'}


    def execute(self, context):
        print('Reset!!')
        return {'FINISHED'}
class Carving(bpy.types.Operator):
    """Carving"""
    bl_idname = "test.carving"
    bl_label = "Carve Surface"
    bl_options = {'REGISTER', 'UNDO'}


    def execute(self, context):
        print('Carving!!')
        return {'FINISHED'}
    @classmethod
    def poll(cls, context):
        return context.object.mode is 'Edit'
    
def register(): 
    bpy.utils.register_class(RandomSample)
    bpy.utils.register_class(Reset)
    bpy.utils.register_class(Carving) 
def unregister(): 
    bpy.utils.unregister_class(RandomSample)
    bpy.utils.unregister_class(Reset)  
    bpy.utils.unregister_class(Carving)
if __name__ == "__main__": 
    register()