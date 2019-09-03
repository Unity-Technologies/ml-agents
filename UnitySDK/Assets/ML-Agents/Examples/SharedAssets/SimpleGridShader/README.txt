This asset is a simple procedural grid shader with GUI customizable parameters.

#Getting Started

After importing the asset, simply drag the material to any object you wish to render grids on. You can also check that the shader is working directly by clicking on the "GridMat" Material.

#Usage

To edit the grid material, the following GUI options are available.

Line Color, Cell Color and Selected Color all represent their respective grid components. Colors use Alpha Cutoff for full transparency by setting the alpha channel to 0.

Grid Size is the amount of cells in the grid. You must make a simple edit to the shader itself to go above 100.

Line Size is the size of lines making up the grid. 

Select Cell enables or disables the cell "Selected" by the parameters Selected Cell X and Y. It only colors a certain cell using a different color, but the base code to "Select" cells in the grid can be used in more complex ways. 

# More information

There is no aspect ratio based scaling by default, to keep cells squared keep the plane mesh square ( or implement your own aspect ratio scaling )

The selected cell can also be chosen by editing the material using the SetFloat value, to potentially make the material interactive. ( See official unity docs to learn more )

The shader, when applied to new materials, can be found under "PDTShaders/TestGrid".

The DistanceScaling.cs script can be used to create a distance based scaling of the line thickness parameter. 
Add it to your camera object or the object representing the point from which the scaling is calculated.
Then link an object that's rendering a grid with the shader in the "Render Object" slot.

"Min Distance" and "Max Distance" are the distance values used to scale the "Scaling Amount" value between 0% ( when the object distance is equal to or below the min distance ) and 100% ( object distance equal or higher to max distance )
This scaled value is then added to the "Line Thickness" value of the shader. A negative value can be set as the "Scaling Amount" to get the inverted effect.

#Support

Thank you for downloading! If you have any problems, questions or bugs to report, you can do so at the following email address : r3eckon@gmail.com