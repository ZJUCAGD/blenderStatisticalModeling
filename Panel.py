import bpy
from bpy.props import FloatProperty, PointerProperty
from bpy.types import PropertyGroup
def sculpture_update(self, context):
    print(context.scene.CarveType)
SamplePropertyGroup = type(
    "SamplePropertyGroup",
    (PropertyGroup,),
    {
        "sigma": FloatProperty(name="σ", default=10.0),
        "x": FloatProperty(name="scale_X", default=1.0),
        "y": FloatProperty(name="scale_Y", default=1.0),
        "z": FloatProperty(name="scale_Z", default=1.0)
    })


class CarvePrepare(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Carving Manager"
    bl_idname = "SCENE_PT_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = "UI"
    bl_category = 'Carving Manager'

    @classmethod
    def poll(cls, context):
        return context.active_object is not None 

    def draw(self, context):
        layout = self.layout

        obj = context.object
        
        row = layout.row()
        row.label(text="Active Object : " + obj.name)
        row = layout.row()
        row.prop(obj, 'name')
        
        row = layout.row()
        row.prop(context.scene, "CarveType", icon="SELECT_DIFFERENCE")
        
        # Create a simple row.
        layout.label(text=" Choose a binary image for sculpting:")
        row = layout.row()
        row.operator("test.open_filebrowser", icon= "FILEBROWSER")
        
        layout.label(text = "Verify image connectivity")
        row = layout.row()
        row.operator("uv.unwrap",icon="FAKE_USER_ON")
        
        row = layout.row()
        row.label(text = "Unwrap the Surface: ")
        row.operator("uv.unwrap", icon ="UV")
        
        row = layout.row()
        row.label(text = "Adapative Subdivide :")
        row.operator("mesh.subdivide", icon ="MESH_GRID")
        
        row = layout.row()
        row.label(text = "Carving Operation : ")
        row.operator("test.carving")
        
        
        row = layout.row()
        row.label(text = "Laplacian Smoothing :")
        row.operator("mesh.vertices_smooth")
class StatisticalShapeModeling(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Statistical Shape Modeling"
    bl_idname = "SCENE_PT_layout2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = "UI"
    bl_category = 'Carving Manager'

    @classmethod
    def poll(cls, context):
        return context.active_object is not None 

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        layout.label(text=" Parameters of Gaussian Kernal:")
        obj = context.object

        sampleProperty = context.scene.samplePropertyGroup
        col = layout.column(align=True)
        col.prop(sampleProperty, "sigma")
        col.prop(sampleProperty, "x")
        col.prop(sampleProperty, "y")
        col.prop(sampleProperty, "z")
        
        row = layout.row()
        row.operator("test.randomsample", icon= "GPBRUSH_RANDOMIZE")
        row.operator("test.reset", icon= "LOOP_BACK")
def register():
    def getsets(self, context):
        sets = [("HOLLOW", "Hollow", "", 1),
    ("YANG SCULPTURE", "Yang Sculpture", "", 2),
    ("YIN SCULPTURE", "Yin Sculpture", "", 3),
        ]
        return sets
    bpy.utils.register_class(SamplePropertyGroup)
    bpy.types.Scene.CarveType = bpy.props.EnumProperty(items=getsets, update=sculpture_update)
    bpy.types.Scene.samplePropertyGroup = PointerProperty(type=SamplePropertyGroup)
    bpy.utils.register_class(CarvePrepare)
    bpy.utils.register_class(StatisticalShapeModeling)

def unregister():
    bpy.utils.register_class(SamplePropertyGroup)
    bpy.utils.unregister_class(CarvePrepare)
    bpy.utils.unregister_class(StatisticalShapeModeling)
    del bpy.types.Scene.samplePropertyGroup
    del bpy.types.Scene.CarveType
if __name__ == "__main__":
    register()