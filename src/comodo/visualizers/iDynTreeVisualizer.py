import idyntree.bindings as iDynTree
import numpy as np
import time 

class iDynTreeVisualizer(): 
    def __init__(self, model_name) -> None:
        self.model_name= model_name
        pass  

    def prepare_visualization(self):
        self.viz = iDynTree.Visualizer()  
        vizOpt = iDynTree.VisualizerOptions()
        vizOpt.winWidth = 1500
        vizOpt.winHeight = 1500 
        self.viz.init(vizOpt)

        self.env = self.viz.enviroment()
        self.env.setElementVisibility('floor_grid',True)
        self.env.setElementVisibility('world_frame',False)
        self.viz.setColorPalette("meshcat")
        self.env.setElementVisibility('world_frame',False)
        self.frames = self.viz.frames()  
        cam = self.viz.camera()
        cam.setPosition(iDynTree.Position(0,3,1.2))
        self.viz.camera().animator().enableMouseControl(True)
    
    def add_model(self,robot_model, urdf_path):
        mdlLoader = iDynTree.ModelLoader()
        mdlLoader.loadReducedModelFromFile(urdf_path,robot_model.joint_name_list, robot_model.base_link)
        self.viz.addModel(mdlLoader.model(),self.model_name)
    
    def update_model(self,s, H_b): 
        s_idyn = self.get_idyntree_joint_pos(s)
        H_b_idyn = self.get_idyntree_transf(H_b)
        self.viz.modelViz(self.model_name).setPositions(H_b_idyn,s_idyn)

    def get_idyntree_transf(self,H): 
        pos_iDynTree = iDynTree.Position()
        R_iDynTree = iDynTree.Rotation()
        R_iDynTree.FromPython(H[:3,:3])
        for i in range(3):
            pos_iDynTree.setVal(i,float(H[i,3]))
            for j in range(3):
                R_iDynTree.setVal(j,i,float(H[j,i]))
        T = iDynTree.Transform()
        T.setPosition(pos_iDynTree)
        T.setRotation(R_iDynTree)
        T.setPosition(pos_iDynTree)
        return T

    def visualize(self, time_step): 
        time_now = time.time()
        while(time.time()-time_now<time_step and self.viz.run()): 
            self.viz.draw()
    def get_idyntree_joint_pos(self,s):
        N_DoF = len(s)
        s_idyntree =iDynTree.VectorDynSize(N_DoF)
        for i in range(N_DoF):
            s_idyntree.setVal(i,s[i])
        return s_idyntree
