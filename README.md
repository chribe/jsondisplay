# NeXus JSON Geometry Viewer 
A browser-based 3D geometry viewer and editor for NeXus instrument descriptions stored in JSON format. The application reads NeXus groups, resolves local `depends_on` transformation chains, calculates global positions, and displays the resulting geometry in an interactive Plotly 3D view. It also provides a resizable JSON structure tree in the same browser window. JSON values can be edited, the geometry can be recalculated, and the modified JSON can be downloaded. 
## Features 
- Interactive 3D Plotly geometry visualisation 
- Automatically detects **all groups with an `NX_class` attribute** 
- No manually maintained list of supported component classes is required. 
- Standard and custom NeXus classes are included. 
- Stable automatically generated colours for each `NX_class` 
- Recursive NeXus `depends_on` transformation resolution 
- Supports `translation` and `rotation` transformations 
- Supports angles in degrees and radians 
- Detector/pixel point-cloud geometry from offset datasets 
- Spatial filtering with X, Y, and Z limits 
- Wildcard-based component exclusion 
- Resizable JSON structure panel 
- Per-component visibility checkboxes 
- GUI controls for applying exclusions, hiding all, showing all, and resetting default exclusions 
- Default exclusions: `*transform*, NXLog` 
- Editable JSON scalar values and small JSON lists 
- **Update geometry** after editing JSON values 
- **Undo** and **Redo** JSON edits - **Save JSON...** browser download of the modified JSON 
- Double-click a 3D point to highlight the corresponding component and open its parent JSON-tree branches 
- A single click in the 3D view does not change the tree 
## Requirements Python 3.9 or newer is recommended. 
Install the required packages: 

```bash pip install numpy plotly dash ``` 

For development or editable installation: 

```bash pip install -e . ``` 

After installation, the viewer may be available as: 

```bash jsondisplay instrument.json ``` 

Otherwise, run it directly as a Python script: 

```bash python jsondisplay.py instrument.json ``` 

## Basic Usage 
```bash python jsondisplay.py instrument.json ``` 
The application starts a local Dash server and automatically opens the viewer in your default browser. 

By default, the viewer URL is: 

```text http://127.0.0.1:8050/ ``` 

The browser page contains a JSON structure and editing panel on the left and an interactive 3D geometry view on the right. The JSON panel width can be adjusted by dragging its right border. 
## Viewer Controls 
### JSON tree 
- Objects and structural lists can be expanded or collapsed. 
- Large numerical arrays are summarised rather than rendered element-by-element. 
- Groups containing `NX_class` have a visibility checkbox. 
- Unchecking a group hides its associated marker and geometry from the 3D view. 
- The tree panel can be resized horizontally. ### Point selection 
- **Single-click** a point in the 3D plot: no action. 
- **Double-click** a component marker or detector point: 
- opens the relevant branch in the JSON tree; 
- highlights the related `NX_class` component. 
### Editing JSON 
Scalar values and small lists can be edited directly in the JSON tree. Examples of editable values: 

```
text values: 1.5 
depends_on: /entry/instrument/detector/transformations/location 
vector: [0.0, 0.0, 1.0] 
units: m 
``` 

Click **Update geometry** to rebuild the 3D geometry using the current JSON data. 

The expanded/collapsed state of the JSON tree is preserved when using this control. > Large numerical arrays, such as detector pixel-offset arrays, are displayed as summaries and are not intended for inline editing. 
### Numeric values 
Numeric editor fields support integers, decimal values, and scientific notation: 

```1.5 10 -0.25 1e-3 -2.5E+4 ```

 Values can change freely between integer and floating-point notation: ```text 1.5 → 10 → 1.5 ``` 
 ### Undo and redo 
 The viewer stores up to 30 JSON states in its edit history. Use **Undo** and **Redo** to restore JSON values and rebuild the geometry. A new edit after an Undo operation clears the redo history. 
 ### Saving modified JSON 
 Click **Save JSON...** to download the modified document through the browser's normal download/save workflow. The download filename is: ``` modified_nexus.json ``` 
 
 The original input file is never overwritten automatically. 
 ## Excluding Components 
 ### GUI exclusions 
 The JSON panel contains an **Exclude from 3D view** field. Patterns can be separated with commas, semicolons, or line breaks: ``` *transform*, NXLog, monitor*, detector_test ``` 
 
 Patterns are matched case-insensitively against component name, `NX_class`, and full NeXus path. Use `*` as a wildcard. Examples: ``` monitor* NXdetector *chopper* */instrument/monitor_1 ``` 
 | Button | Function | 
 |---|---| 
 | **Apply exclusions** | Hides all components matching the current patterns | 
 | **Hide all** | Hides every plotted component | 
 | **Show all** | Shows every component and clears the exclusion field | 
 | **Reset** | Restores the default exclusions | 
 
 The default exclusion patterns are: ```text *transform*, NXLog ``` 
 ### Command-line exclusions 
 Additional patterns may be supplied through `--exclude`: 
 ```bash 
 python jsondisplay.py instrument.json --exclude "monitor*" "NXdetector" 
 ```
  These patterns are added to the initial GUI exclusion patterns. 
  ## Spatial Filtering 
  Geometry can be restricted to a global coordinate range: 
  | Option | Description | 
  |---|---| 
  | `--xmin` | Minimum X coordinate | 
  | `--xmax` | Maximum X coordinate | 
  | `--ymin` | Minimum Y coordinate | 
  | `--ymax` | Maximum Y coordinate | 
  | `--zmin` | Minimum Z coordinate | 
  | `--zmax` | Maximum Z coordinate | 
  
  All limits are optional. Example, showing geometry within a Z range: 
  ```bash 
  python jsondisplay.py instrument.json --zmin -100 --zmax 100 
  ```
Example, showing geometry in a three-dimensional bounding box: 
```bash 
python jsondisplay.py instrument.json --xmin -50 --xmax 50 --ymin -20 --ymax 20  --zmin -100 --zmax 100 
``` 
Spatial limits are applied after NeXus transformation chains have been resolved, so they refer to global coordinates displayed by the viewer. 
## Large Detector Geometries
 Detector geometry can contain very large numbers of pixel positions. To keep the browser responsive, point clouds can be downsampled: 
 ```bash 
 python jsondisplay.py instrument.json --max-points 20000 
 ```
  The default is: ``` 100000 points per component ``` 
  
  For large files, values between `10,000` and `50,000` are often more responsive. Spatial filtering can also substantially reduce the number of rendered points. 
  ## NeXus Transformations 
  The viewer supports: - `translation` - `rotation` Values with units `radians` or `rad` are converted to degrees before constructing rotation matrices. Transformation chains are resolved recursively through the NeXus `depends_on` relationship: ```transform_3 depends_on -> transform_2 transform_2 depends_on -> transform_1 transform_1 depends_on -> . ``` Components with no resolvable transformations use the identity matrix and appear at the origin. 
  ## Detector and Pixel Geometry 
  The viewer recognises the following offset dataset combinations: ``` x_pixel_offset y_pixel_offset z_pixel_offset ``` and: ``` x_offset y_offset z_offset ``` 
  
  Local points are transformed using the component transformation matrix and displayed as a 3D point cloud. Spatial filtering is applied to the transformed global points. 
  ## Command-Line 
  Reference Run: 
  ```bash 
  python jsondisplay.py --help 
  ``` 
  
  General syntax: ``` python jsondisplay.py FILE [OPTIONS] ``` 
  
  Available options include: ``` --exclude PATTERN [PATTERN ...] --xmin VALUE --xmax VALUE --ymin VALUE --ymax VALUE --zmin VALUE --zmax VALUE --max-points VALUE --port VALUE ``` 
  
  Example: 
  ```bash 
  python jsondisplay.py instrument.json  --exclude "monitor*" "test*"   --xmin -50  --xmax 50   --zmin -100   --zmax 100   --max-points 25000 
  ```

## Output
The viewer provides: 
- 3D rotation, panning, and zooming; 
- component labels; 
- hover information with component name, class, path, and position; 
- automatically colour-coded NeXus classes;
- detector/pixel point clouds; 
- component visibility controls; 
- JSON editing and browser download of modified JSON; 
- double-click navigation from a 3D point to the matching JSON-tree component. Axes are labelled: ``` X [m] Y [m] Z [m] ``` 
## Notes 
- All groups with an `NX_class` attribute are detected automatically. 
- The JSON tree summarises large arrays to avoid creating excessive browser elements. 
- The application runs locally at `127.0.0.1`; it does not upload the JSON file to an external service. 
- Browser download is used for saving edited JSON to avoid silently overwriting the original file. 
- AI-assisted code generation has been used in the development of this project. 