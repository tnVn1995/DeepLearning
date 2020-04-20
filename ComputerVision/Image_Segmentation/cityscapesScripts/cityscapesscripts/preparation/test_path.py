#%%
import os
import glob
path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
root = r'''\personal_projects\DeepLearning\ComputerVision\Image_Segmentation\cityscapesScripts\cityscapesscripts\preparation\..\..'''
searchFine   = os.path.join( root , "gtFine"   , "*" , "*" , "*_gt*_polygons.json" )

glob.glob( searchFine )



path


#%%
if __name__ == '__main__':
    print(path)
